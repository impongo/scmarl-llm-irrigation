import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Optional, Sequence
from paper_prompt import DECISION_SYSTEM_MESSAGE, EXPLANATION_SYSTEM_MESSAGE, GateHistory, build_paper_decision_prompt
import atexit
import time

from core_scmarl_vs_dmarl.scheduler_evaluation_exp6_llm_per_zone_scmarl_local import (
    SolveScheduler,
    configure_llm,
)
from llm_explainability import (
    ExplanationRecorder,
    build_per_zone_explanation_prompt,
    parse_per_zone_explanation,
    validate_per_zone_explanation,
)
from season_long_runner_exp456 import run_season_long_experiment

_PAPER_GATE_HISTORY = GateHistory()

EXPERIMENT_NAME = "Experiment 6 - SCMARL-LLM (Per-Zone 0/1) - SCMARL local agents"
RESULTS_DIR_NAME = "results_exp6_llm_per_zone_scmarl_local"

LLM_BACKEND_ENABLED = True
_FALLBACK_WARNED = False
# API调用策略:
# - "once_per_day": 每个仿真日只调1次LLM（更快，推荐）
# - "per_step":     每天按预测步长调多次LLM（原始实现）
LLM_CALL_POLICY = "once_per_day"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "qwen3.5:27b"
LLM_TIMEOUT_SEC = 90

LLM_PROVIDER = "dashscope"  # "dashscope" or "ollama"
DASHSCOPE_URL = os.getenv("DASHSCOPE_API_URL", "").strip()
DASHSCOPE_MODEL = "qwen3.5-27b"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_TIMEOUT_SEC = 180
LLM_MAX_RETRIES = 4
LLM_RETRY_SLEEP_SEC = 2.0
ENABLE_EXPLANATION_MODULE = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DAY_COUNTER = 0

_EXPLANATION_RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR_NAME)
_EXPLANATION_RECORDER = ExplanationRecorder(
    results_dir=_EXPLANATION_RESULTS_DIR,
    experiment_name=EXPERIMENT_NAME,
    mode="per_zone",
)
atexit.register(_EXPLANATION_RECORDER.write_summary)


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
def _stage_sensitivity_from_kc(kc: float):
    if kc < 0.30:
        return "EMERGENCE", "LOW"
    if kc < 0.55:
        return "TILLERING", "MEDIUM"
    if kc < 0.85:
        return "JOINTING", "HIGH"
    if kc < 1.15:
        return "HEADING_ANTHESIS", "CRITICAL"
    return "GRAIN_FILLING", "HIGH"


def _build_trace(day: int, context: Dict, decisions: Sequence[float]) -> Dict:
    rz = [float(x) for x in context["root_zone_moisture"]]
    lb = [float(x) for x in context["lower_bounds"]]
    margin = [rz[i] - lb[i] for i in range(3)]
    kc = float(context["current_inputs"]["crop_coeff"])
    stage, sensitivity = _stage_sensitivity_from_kc(kc)
    fut_rain = [float(x) for x in context["future_inputs"]["rain_mm_per_day"]]
    fut_et = [float(x) for x in context["future_inputs"]["ref_et_mm_per_day"]]
    fut_kc = [float(x) for x in context["future_inputs"]["crop_coeff"]]
    n_fut = min(len(fut_rain), len(fut_et), len(fut_kc))

    def window(days: int):
        n = min(days, n_fut)
        r = sum(fut_rain[:n])
        d = sum(fut_et[i] * fut_kc[i] for i in range(n))
        return r, d

    rain_3d, etc_3d = window(3)
    sdr_3d = rain_3d / etc_3d if etc_3d > 0.01 else 99.0
    rain_tmr = fut_rain[1] if n_fut > 1 else 0.0
    etc_tmr = fut_et[1] * fut_kc[1] if n_fut > 1 else float(context["current_inputs"]["crop_coeff"]) * float(
        context["current_inputs"]["ref_et_mm_per_day"]
    )

    return {
        "day": int(day),
        "stage": stage,
        "sensitivity": sensitivity,
        "margin_mz1": float(margin[0]),
        "margin_mz2": float(margin[1]),
        "margin_mz3": float(margin[2]),
        "min_margin": float(min(margin)),
        "rain_3d": float(rain_3d),
        "etc_3d": float(etc_3d),
        "sdr_3d": float(sdr_3d),
        "rain_tomorrow": float(rain_tmr),
        "etc_tomorrow": float(etc_tmr),
        "allow_mz1": int(1 if float(decisions[0]) >= 0.5 else 0),
        "allow_mz2": int(1 if float(decisions[1]) >= 0.5 else 0),
        "allow_mz3": int(1 if float(decisions[2]) >= 0.5 else 0),
    }


def call_llm_explainer_exp6(prompt: str) -> Optional[str]:
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            if LLM_PROVIDER == "dashscope":
                api_key = DASHSCOPE_API_KEY.strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
                if not api_key or not DASHSCOPE_URL:
                    return None
                payload = {
                    "model": DASHSCOPE_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": EXPLANATION_SYSTEM_MESSAGE,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "stream": False,
                }
                req = urllib.request.Request(
                    DASHSCOPE_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                timeout = DASHSCOPE_TIMEOUT_SEC
            else:
                payload = {
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "top_p": 0.9, "num_predict": 180},
                }
                req = urllib.request.Request(
                    OLLAMA_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                timeout = LLM_TIMEOUT_SEC

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw)
            if "error" in obj:
                raise RuntimeError(f"backend returned error: {obj['error']}")

            if LLM_PROVIDER == "dashscope":
                choices = obj.get("choices", [])
                if not choices:
                    raise RuntimeError("empty choices")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = "".join(text_parts)
                text = content
            else:
                text = obj.get("response", "")
                if not text and isinstance(obj.get("message"), dict):
                    text = obj["message"].get("content") or ""

            if isinstance(text, str) and text.strip():
                return text.strip()
            raise RuntimeError("empty response text")
        except Exception as e:
            last_error = str(e)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP_SEC * attempt)
    print(f"[WARN] LLM explainer failed after {LLM_MAX_RETRIES} attempts: {last_error}")
    return None


def call_llm_backend_exp6(prompt: str) -> Optional[str]:
    if LLM_PROVIDER == "dashscope":
        api_key = DASHSCOPE_API_KEY.strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key or not DASHSCOPE_URL:
            print("[WARN] Set DASHSCOPE_API_KEY and DASHSCOPE_API_URL environment variables.")
            return None

        payload = {
            "model": DASHSCOPE_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": DECISION_SYSTEM_MESSAGE,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "stream": False,
        }
        req = urllib.request.Request(
            DASHSCOPE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        timeout = DASHSCOPE_TIMEOUT_SEC
    else:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 120,
            },
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = LLM_TIMEOUT_SEC

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        obj = json.loads(raw)

        if "error" in obj:
            print(f"[WARN] LLM backend returned error: {obj['error']}")
            return None

        if LLM_PROVIDER == "dashscope":
            choices = obj.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "".join(text_parts)
            text = content
        else:
            text = obj.get("response", "")
            if not text:
                text = obj.get("thinking", "")
            if not text and isinstance(obj.get("message"), dict):
                text = obj["message"].get("content") or obj["message"].get("thinking") or ""

        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[WARN] LLM backend HTTP {e.code}: {body[:500]}")
        return None
    except Exception as e:
        print(f"[WARN] LLM backend error: {e}")
        return None


def parse_llm_output_exp6(raw_text: str) -> Optional[Sequence[float]]:
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()
    if not text:
        return None

    text = text.replace("```json", "```").replace("```JSON", "```")
    if "```" in text:
        parts = [p.strip() for p in text.split("```") if p.strip()]
        for p in parts:
            if p.startswith("{") and p.endswith("}"):
                text = p
                break

    m_obj = re.search(r"\{[\s\S]*\}", text)
    candidate = m_obj.group(0) if m_obj else text

    try:
        data = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    keys = ["allow_mz1", "allow_mz2", "allow_mz3"]
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
            if v in ("1", "true", "yes", "allow"):
                decisions.append(1.0)
            elif v in ("0", "false", "no", "deny"):
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
    global _FALLBACK_WARNED, _DAY_COUNTER
    _DAY_COUNTER += 1
    sim_day = _DAY_COUNTER

    prompt = build_prompt_exp6(context)
    raw_response = call_llm_backend_exp6(prompt) if LLM_BACKEND_ENABLED else None
    decision = parse_llm_output_exp6(raw_response) if raw_response else None
    if decision is not None:
        _PAPER_GATE_HISTORY.record(sim_day, any(float(x) >= 0.5 for x in decision))

    if decision is None and not _FALLBACK_WARNED:
        print("[WARN] Experiment 6 did not receive a valid LLM decision. Falling back to rule coordinator.")
        _FALLBACK_WARNED = True

    if decision is not None and ENABLE_EXPLANATION_MODULE:
        try:
            trace = _build_trace(sim_day, context, decision)
            explain_prompt = build_per_zone_explanation_prompt(trace)
            explain_raw = call_llm_explainer_exp6(explain_prompt)
            explain_obj = parse_per_zone_explanation(explain_raw)
            validation = validate_per_zone_explanation(trace, explain_obj)
            _EXPLANATION_RECORDER.append(
                day=sim_day,
                trace=trace,
                decision_raw_text=raw_response,
                explanation_raw_text=explain_raw,
                parsed_explanation=explain_obj,
                validation=validation,
            )
            _EXPLANATION_RECORDER.write_summary()
        except Exception as e:
            print(f"[WARN] Explanation module failed on day {sim_day}: {e}")
    return decision


def _reset_explainability_state() -> None:
    global _DAY_COUNTER
    _DAY_COUNTER = 0
    _PAPER_GATE_HISTORY.reset()
    for path in (_EXPLANATION_RECORDER.records_path, _EXPLANATION_RECORDER.summary_path):
        if os.path.exists(path):
            os.remove(path)
    _EXPLANATION_RECORDER.stats = {
        "total": 0,
        "parse_ok": 0,
        "decision_match": 0,
        "fields_ok": 0,
        "evidence_keys_ok": 0,
        "confidence_ok": 0,
        "data_sanity_ok": 0,
        "overall_ok": 0,
    }


if __name__ == "__main__":
    _reset_explainability_state()
    configure_llm(callback=llm_per_zone_callback, call_policy=LLM_CALL_POLICY)
    run_season_long_experiment(
        experiment_name=EXPERIMENT_NAME,
        solve_scheduler=SolveScheduler,
        results_dir_name=RESULTS_DIR_NAME,
    )