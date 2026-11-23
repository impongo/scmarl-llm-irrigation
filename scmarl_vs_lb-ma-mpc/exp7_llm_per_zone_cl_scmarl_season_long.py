import contextlib
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import urllib.request
from typing import Dict, Optional, Sequence
import numpy as np
import scmarl_season_long as cl_scmarl_runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, 'core_scmarl_vs_lb-mpc')
PAPER_PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scmarl_vs_dmarl"))
if PAPER_PROMPT_DIR not in sys.path:
    sys.path.insert(0, PAPER_PROMPT_DIR)
from paper_prompt import DECISION_SYSTEM_MESSAGE, EXPLANATION_SYSTEM_MESSAGE, GateHistory, build_paper_decision_prompt
if CORE_DIR not in os.sys.path:
    os.sys.path.append(CORE_DIR)

from scheduler_formulation_exp7_llm_per_zone_scmarl_local import SolveScheduler, configure_llm

_PAPER_GATE_HISTORY = GateHistory()

EXPERIMENT_NAME = 'Experiment 7 - CL-SCMARL LLM Per-Zone (0/1) - SCMARL local agents'
RESULTS_DIR_NAME = 'result-llm-scmarl-exp7'

LLM_BACKEND_ENABLED = True
_FALLBACK_WARNED = False
# API调用策略:
# - "once_per_day": 每个仿真日只调1次LLM（更快，推荐）
# - "per_step":     每天按预测步长调多次LLM（原始实现）
LLM_CALL_POLICY = 'once_per_day'

OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
LLM_MODEL = 'qwen-plus'
LLM_TIMEOUT_SEC = 90

LLM_PROVIDER = 'dashscope'  # 'dashscope' or 'ollama'
DASHSCOPE_URL = os.getenv("DASHSCOPE_API_URL", "").strip()
DASHSCOPE_MODEL = "qwen3.5-27b"
# Read DASHSCOPE_API_KEY from the environment; do not put credentials in this file.
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_TIMEOUT_SEC = 180

_BASE_RESULTS_DIR = os.path.join(BASE_DIR, 'results_cl-scmarl')
_TARGET_RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR_NAME)
_RUN_LOG_DIR = os.path.join(BASE_DIR, 'run_logs')


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _evaluate_water_stress(theta_v: float, lower_zone: float, upper_zone: float, pwp: float, anaerobic: float) -> float:
    if theta_v <= pwp:
        return 0.0
    if pwp < theta_v <= lower_zone:
        return (theta_v - pwp) / (lower_zone - pwp)
    if lower_zone < theta_v < upper_zone:
        return 1.0
    if upper_zone <= theta_v <= anaerobic:
        return (theta_v - upper_zone) / (anaerobic - upper_zone)
    return 0.0


def _print_metrics_summary(result_dir: str) -> None:
    try:
        actual_field_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'common', 'actual_field'))
        if actual_field_dir not in sys.path:
            sys.path.insert(0, actual_field_dir)

        from crop_coefficient_calculation import generate_Kc  # type: ignore
        from parameters_mpc import ManagementZone_1, ManagementZone_2, ManagementZone_3  # type: ignore
        from van_Genuchten_relations import volumetricMoisture  # type: ignore

        weather_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'common', 'weather_data'))
        avg_temp = np.atleast_1d(np.loadtxt(os.path.join(weather_dir, 'dAVG_TEMP.txt')))
        crop_coeff = np.atleast_1d(generate_Kc(avg_temp))
        ref_evap = np.atleast_1d(np.loadtxt(os.path.join(weather_dir, 'dPET.txt')))
        ref_evap = np.clip(ref_evap, 1.04, 9.0)

        soil_1 = ManagementZone_1()
        soil_2 = ManagementZone_2()
        soil_3 = ManagementZone_3()

        lower = [0.200, 0.200, 0.230]
        upper = [0.281, 0.281, 0.301]
        pwp = [
            float(volumetricMoisture(-152.85, soil_1)),
            float(volumetricMoisture(-152.85, soil_2)),
            float(volumetricMoisture(-152.85, soil_3)),
        ]
        anaerobic = [
            float(volumetricMoisture(-0.1, soil_1)),
            float(volumetricMoisture(-0.1, soil_2)),
            float(volumetricMoisture(-0.1, soil_3)),
        ]

        y_max = 0.88  # kg/m^2
        ky = 1.15

        zone_yields = []
        zone_irrig = []
        zone_iwue = []

        for idx, mz in enumerate((1, 2, 3)):
            state_path = os.path.join(result_dir, f'StateCL_mz{mz}.txt')
            input_path = os.path.join(result_dir, f'InputCL_mz{mz}.txt')
            state = np.atleast_1d(np.loadtxt(state_path))
            irrigation_input = np.atleast_1d(np.loadtxt(input_path))

            n_points = min(len(state), len(crop_coeff), len(ref_evap))
            max_et = crop_coeff[:n_points] * ref_evap[:n_points]
            seasonal_et = 0.0
            for t in range(n_points):
                ws = _evaluate_water_stress(float(state[t]), lower[idx], upper[idx], pwp[idx], anaerobic[idx])
                seasonal_et += ws * float(max_et[t])

            denom = float(np.sum(max_et))
            pred_yield = float((1.0 - ky + ky * (seasonal_et / denom)) * y_max) if denom > 0 else float('nan')
            irrigation = -float(np.sum(irrigation_input) / 1000.0)
            iwue = float(pred_yield / irrigation) if irrigation > 1e-12 else float('nan')

            zone_yields.append(pred_yield)
            zone_irrig.append(irrigation)
            zone_iwue.append(iwue)

        field_yield = float(np.mean(zone_yields))
        field_irrig = float(np.sum(zone_irrig))
        field_iwue = float(field_yield / field_irrig) if field_irrig > 1e-12 else float('nan')

        print('\n================ Metrics Summary ================')
        print(f'Experiment: {EXPERIMENT_NAME}')
        print(f'Results folder: {result_dir}')
        print(
            'Yield (kg/m^2) => '
            f'MZ1: {zone_yields[0]:.4f}, MZ2: {zone_yields[1]:.4f}, MZ3: {zone_yields[2]:.4f}, AVG: {field_yield:.4f}'
        )
        print(
            'Total irrigation (m^3/m^2) => '
            f'MZ1: {zone_irrig[0]:.4f}, MZ2: {zone_irrig[1]:.4f}, MZ3: {zone_irrig[2]:.4f}, SUM: {field_irrig:.4f}'
        )
        print(
            'IWUE (kg/m^3) => '
            f'MZ1: {zone_iwue[0]:.4f}, MZ2: {zone_iwue[1]:.4f}, MZ3: {zone_iwue[2]:.4f}, Overall: {field_iwue:.4f}'
        )
        print('=================================================\n')
    except Exception as exc:
        print(f'[WARN] Failed to compute summary metrics: {exc}')



def build_prompt_exp6(context: Dict) -> str:
    day = int(context.get("sim_day", context.get("step_index", 0) + 1))
    rz = [float(x) for x in context["root_zone_moisture"]]
    lb = [float(x) for x in context["lower_bounds"]]
    kc = float(context["current_inputs"]["crop_coeff"])
    pet = float(context["current_inputs"]["ref_et_mm_per_day"])
    lai = float(context["current_inputs"]["lai"])
    etc_today = kc * pet
    if kc < 0.30:
        stage, sensitivity = "EMERGENCE", "LOW"
    elif kc < 0.55:
        stage, sensitivity = "TILLERING", "MEDIUM"
    elif kc < 0.85:
        stage, sensitivity = "JOINTING", "HIGH"
    elif kc < 1.15:
        stage, sensitivity = "HEADING_ANTHESIS", "CRITICAL"
    else:
        stage, sensitivity = "GRAIN_FILLING", "HIGH"
    margins = [rz[i] - lb[i] for i in range(min(len(rz), len(lb)))]
    fut_rain = [float(x) for x in context["future_inputs"]["rain_mm_per_day"]]
    fut_et = [float(x) for x in context["future_inputs"]["ref_et_mm_per_day"]]
    fut_kc = [float(x) for x in context["future_inputs"]["crop_coeff"]]
    n = min(3, len(fut_rain), len(fut_et), len(fut_kc))
    rain_3d = sum(fut_rain[:n])
    etc_3d = sum(fut_et[i] * fut_kc[i] for i in range(n))
    sdr_3d = rain_3d / etc_3d if etc_3d > 0.01 else 99.0
    rain_tomorrow = fut_rain[1] if len(fut_rain) > 1 else 0.0
    etc_tomorrow = fut_et[1] * fut_kc[1] if n > 1 else etc_today
    history = _PAPER_GATE_HISTORY.features(day)
    return build_paper_decision_prompt(
        crop="wheat", day=day, stage=stage, sensitivity=sensitivity,
        kc=kc, pet=pet, etc_today=etc_today, lai=lai, margins=margins,
        sdr_3d=sdr_3d, rain_tomorrow=rain_tomorrow,
        etc_tomorrow=etc_tomorrow, history=history, per_zone=True)
def call_llm_backend_exp6(prompt: str) -> Optional[str]:
    if LLM_PROVIDER == 'dashscope':
        api_key = DASHSCOPE_API_KEY.strip() or os.getenv('DASHSCOPE_API_KEY', '').strip()
        if not api_key or not DASHSCOPE_URL:
            print("[WARN] Set DASHSCOPE_API_KEY and DASHSCOPE_API_URL environment variables.")
            return None

        payload = {
            'model': DASHSCOPE_MODEL,
            'messages': [
                {
                    'role': 'system',
                    'content': DECISION_SYSTEM_MESSAGE,
                },
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0,
            'stream': False,
        }
        req = urllib.request.Request(
            DASHSCOPE_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        timeout = DASHSCOPE_TIMEOUT_SEC
    else:
        payload = {
            'model': LLM_MODEL,
            'prompt': prompt,
            'stream': False,
            'format': 'json',
            'options': {
                'temperature': 0,
                'top_p': 0.9,
                'num_predict': 120,
            },
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        timeout = LLM_TIMEOUT_SEC

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='ignore')
        obj = json.loads(raw)

        if 'error' in obj:
            print(f"[WARN] LLM backend returned error: {obj['error']}")
            return None

        if LLM_PROVIDER == 'dashscope':
            choices = obj.get('choices', [])
            if not choices:
                return None
            message = choices[0].get('message', {})
            content = message.get('content', '')
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content = ''.join(text_parts)
            text = content
        else:
            text = obj.get('response', '')
            if not text:
                text = obj.get('thinking', '')
            if not text and isinstance(obj.get('message'), dict):
                text = obj['message'].get('content') or obj['message'].get('thinking') or ''

        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except Exception as e:
        print(f'[WARN] LLM backend error: {e}')
        return None


def parse_llm_output_exp6(raw_text: str) -> Optional[Sequence[float]]:
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()
    if not text:
        return None

    text = text.replace('```json', '```').replace('```JSON', '```')
    if '```' in text:
        parts = [p.strip() for p in text.split('```') if p.strip()]
        for p in parts:
            if p.startswith('{') and p.endswith('}'):
                text = p
                break

    m_obj = re.search(r'\{[\s\S]*\}', text)
    candidate = m_obj.group(0) if m_obj else text

    try:
        data = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    keys = ['allow_mz1', 'allow_mz2', 'allow_mz3']
    if any(k not in data for k in keys):
        return None

    decisions = []
    for key in keys:
        value = data[key]
        if isinstance(value, bool):
            decisions.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            decisions.append(1.0 if float(value) >= 0.5 else 0.0)
        elif isinstance(value, str):
            v = value.strip().lower()
            if v in ('1', 'true', 'yes', 'allow'):
                decisions.append(1.0)
            elif v in ('0', 'false', 'no', 'deny'):
                decisions.append(0.0)
            else:
                try:
                    decisions.append(1.0 if float(v) >= 0.5 else 0.0)
                except Exception:
                    return None
        else:
            return None

    return decisions


def llm_per_zone_callback(context: Dict) -> Optional[Sequence[float]]:
    global _FALLBACK_WARNED
    prompt = build_prompt_exp6(context)
    raw_response = call_llm_backend_exp6(prompt) if LLM_BACKEND_ENABLED else None
    decision = parse_llm_output_exp6(raw_response) if raw_response else None
    if decision is not None:
        day = int(context.get("sim_day", context.get("step_index", 0) + 1))
        _PAPER_GATE_HISTORY.record(day, any(float(x) >= 0.5 for x in decision))

    if decision is None and not _FALLBACK_WARNED:
        print('[WARN] Experiment 7 did not receive a valid LLM decision. Falling back to rule coordinator.')
        _FALLBACK_WARNED = True
    return decision


def _run_and_capture_results():
    backup_dir = None
    if os.path.exists(_BASE_RESULTS_DIR):
        backup_dir = _BASE_RESULTS_DIR + f'.backup_{int(time.time())}'
        os.replace(_BASE_RESULTS_DIR, backup_dir)

    run_error = None
    partial_results_dir = None
    run_id = time.strftime('%Y%m%d-%H%M%S')
    os.makedirs(_RUN_LOG_DIR, exist_ok=True)
    run_log_path = os.path.join(_RUN_LOG_DIR, f'{RESULTS_DIR_NAME}_{run_id}.log')
    original_cwd = os.getcwd()

    try:
        with open(run_log_path, 'a', encoding='utf-8') as run_log_file:
            tee = _Tee(sys.__stdout__, run_log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(f'[INFO] Run ID: {run_id}')
                print(f'[INFO] Working directory switched to: {BASE_DIR}')

                # scmarl_season_long.py writes plots/files to ./results_cl-scmarl using
                # relative paths, so run from BASE_DIR and ensure directory exists.
                os.chdir(BASE_DIR)
                os.makedirs(_BASE_RESULTS_DIR, exist_ok=True)

                cl_scmarl_runner.main()

                if os.path.exists(_TARGET_RESULTS_DIR):
                    shutil.rmtree(_TARGET_RESULTS_DIR)
                if os.path.exists(_BASE_RESULTS_DIR):
                    os.replace(_BASE_RESULTS_DIR, _TARGET_RESULTS_DIR)
                    print(f'[INFO] Results moved to: {_TARGET_RESULTS_DIR}')
                    _print_metrics_summary(_TARGET_RESULTS_DIR)
                else:
                    print(f'[WARN] Expected results folder not found: {_BASE_RESULTS_DIR}')
    except Exception as exc:
        run_error = exc
        if os.path.exists(_BASE_RESULTS_DIR):
            partial_results_dir = os.path.join(BASE_DIR, f'{RESULTS_DIR_NAME}.failed_{run_id}')
            if os.path.exists(partial_results_dir):
                shutil.rmtree(partial_results_dir, ignore_errors=True)
            shutil.copytree(_BASE_RESULTS_DIR, partial_results_dir)
            print(f'[WARN] Run failed; partial results saved to: {partial_results_dir}')
    finally:
        os.chdir(original_cwd)
        if backup_dir and os.path.exists(backup_dir):
            if os.path.exists(_BASE_RESULTS_DIR):
                shutil.rmtree(_BASE_RESULTS_DIR, ignore_errors=True)
            os.replace(backup_dir, _BASE_RESULTS_DIR)

    # Copy run log into final artifact folder for post-mortem traceability.
    log_target_dir = _TARGET_RESULTS_DIR if run_error is None else partial_results_dir
    if log_target_dir and os.path.exists(log_target_dir):
        shutil.copy2(run_log_path, os.path.join(log_target_dir, 'run.log'))

    if run_error is not None:
        raise run_error


def main():
    print(f'[INFO] Running {EXPERIMENT_NAME}')
    cl_scmarl_runner.SolveScheduler = SolveScheduler
    configure_llm(callback=llm_per_zone_callback, call_policy=LLM_CALL_POLICY)
    _run_and_capture_results()
    print(f'[INFO] Results saved to: {_TARGET_RESULTS_DIR}')


if __name__ == '__main__':
    mp.freeze_support()
    main()