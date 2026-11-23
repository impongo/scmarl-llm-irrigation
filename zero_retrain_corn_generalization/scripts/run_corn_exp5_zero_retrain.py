import json
import os
import re
import sys
import time
import atexit
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, Optional

from corn_overrides import CORN_PARAMS, apply_corn_overrides


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PERF_DIR = os.path.dirname(ROOT_DIR)
BASELINE_DIR = os.path.join(PERF_DIR, "scmarl_vs_dmarl")
CORE_DIR = os.path.join(BASELINE_DIR, "core_scmarl_vs_dmarl")

os.chdir(BASELINE_DIR)
if BASELINE_DIR not in sys.path:
    sys.path.append(BASELINE_DIR)
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from paper_prompt import DECISION_SYSTEM_MESSAGE, EXPLANATION_SYSTEM_MESSAGE, build_paper_decision_prompt

import season_long_runner_exp456 as runner
from scheduler_evaluation_exp5_llm_sync_scmarl_local import SolveScheduler, configure_llm
from llm_explainability import (
    ExplanationRecorder,
    build_global_explanation_prompt,
    parse_global_explanation,
    validate_global_explanation,
)


LLM_BACKEND_ENABLED = True
LLM_CALL_POLICY = "once_per_day"  # or "per_step"

LLM_PROVIDER = "dashscope"  # "dashscope" or "ollama"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen-plus"
OLLAMA_TIMEOUT_SEC = 90

DASHSCOPE_URL = os.getenv("DASHSCOPE_API_URL", "").strip()
DASHSCOPE_MODEL = "qwen3.5-27b"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_TIMEOUT_SEC = 240

LLM_MAX_RETRIES = 4
LLM_RETRY_SLEEP_SEC = 2.0
STRICT_NO_RULE_FALLBACK = True
ENABLE_EXPLANATION_MODULE = True

_FALLBACK_WARNED = False
_SEASON_DAY_COUNTER = 0
_LAST_OPEN_DAY = -999
_OPEN_COUNT_7D = []
_OPEN_COUNT_14D = []
_EXPLANATION_RECORDER: Optional[ExplanationRecorder] = None
_CURRENT_RESULTS_DIR: Optional[str] = None


def _build_stage_from_day(day: int, stage_days: Dict[str, int]) -> str:
    d1 = stage_days["initial"]
    d2 = stage_days["development"]
    d3 = stage_days["mid"]
    if day <= d1:
        return "INITIAL"
    if day <= d1 + d2:
        return "DEVELOPMENT"
    if day <= d1 + d2 + d3:
        return "MID"
    return "LATE"


def _history_features(current_day: int) -> Dict[str, int]:
    global _OPEN_COUNT_7D, _OPEN_COUNT_14D
    _OPEN_COUNT_7D = [d for d in _OPEN_COUNT_7D if current_day - d < 7]
    _OPEN_COUNT_14D = [d for d in _OPEN_COUNT_14D if current_day - d < 14]
    days_since = 999 if _LAST_OPEN_DAY < 0 else (current_day - _LAST_OPEN_DAY)
    return {
        "days_since_last_open": int(days_since),
        "opens_last_7d": int(len(_OPEN_COUNT_7D)),
        "opens_last_14d": int(len(_OPEN_COUNT_14D)),
    }


def build_prompt_exp5_corn(context: Dict) -> str:
    day = int(context.get("sim_day", context.get("step_index", 0) + 1))
    rz = [float(x) for x in context["root_zone_moisture"]]
    lb = [float(x) for x in context["lower_bounds"]]
    kc = float(context["current_inputs"]["crop_coeff"])
    pet = float(context["current_inputs"]["ref_et_mm_per_day"])
    lai = float(context["current_inputs"]["lai"])
    etc_today = kc * pet
    stage = _build_stage_from_day(day, CORN_PARAMS["stage_days"])
    sensitivity = ("CRITICAL" if stage == "MID" else ("HIGH" if stage == "DEVELOPMENT" else ("MEDIUM" if stage == "LATE" else "LOW")))
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
    history = context.get("history", {})
    return build_paper_decision_prompt(
        crop="corn", day=day, stage=stage, sensitivity=sensitivity,
        kc=kc, pet=pet, etc_today=etc_today, lai=lai, margins=margins,
        sdr_3d=sdr_3d, rain_tomorrow=rain_tomorrow,
        etc_tomorrow=etc_tomorrow, history=history, per_zone=False)
def call_llm_backend_corn(prompt: str) -> Optional[str]:
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
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
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "top_p": 0.9, "num_predict": 120},
                }
                req = urllib.request.Request(
                    OLLAMA_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                timeout = OLLAMA_TIMEOUT_SEC

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw)

            if "error" in obj:
                last_error = f"backend returned error: {obj['error']}"
                raise RuntimeError(last_error)

            if LLM_PROVIDER == "dashscope":
                choices = obj.get("choices", [])
                if not choices:
                    last_error = "empty choices"
                    raise RuntimeError(last_error)
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = "".join(text_parts)
                text = content
                if (not text or not str(text).strip()) and isinstance(message, dict):
                    # Some providers put JSON-like text in alternative fields.
                    text = message.get("reasoning_content", "") or message.get("tool_content", "") or ""
                if (not text or not str(text).strip()) and isinstance(choices[0], dict):
                    text = choices[0].get("text", "") or ""
            else:
                text = obj.get("response", "")
                if not text and isinstance(obj.get("message"), dict):
                    text = obj["message"].get("content") or ""

            if isinstance(text, str) and text.strip():
                return text.strip()

            last_error = "empty response text"
            raise RuntimeError(last_error)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            last_error = f"HTTP {e.code}: {body[:300]}"
            print(f"[WARN] LLM backend HTTP error (attempt {attempt}/{LLM_MAX_RETRIES}): {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] LLM backend error (attempt {attempt}/{LLM_MAX_RETRIES}): {last_error}")

        if attempt < LLM_MAX_RETRIES:
            time.sleep(LLM_RETRY_SLEEP_SEC * attempt)

    print(f"[WARN] LLM backend failed after {LLM_MAX_RETRIES} attempts: {last_error}")
    return None


def parse_llm_output_corn(raw_text: str) -> Optional[float]:
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
        m = re.search(r"allow[_\s-]*irrigation[^01tf]*(0|1|true|false)", text, re.IGNORECASE)
        if not m:
            return None
        v = m.group(1).lower()
        if v in ("1", "true"):
            return 1.0
        if v in ("0", "false"):
            return 0.0
        return None

    if not isinstance(data, dict):
        return None

    value = data.get(
        "allow_irrigation",
        data.get("allow", data.get("decision", data.get("global_allow", data.get("valve_decision", None)))),
    )
    if value is None:
        return None

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 0.5 else 0.0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "allow"):
            return 1.0
        if v in ("0", "false", "no", "deny"):
            return 0.0
        try:
            return 1.0 if float(v) >= 0.5 else 0.0
        except Exception:
            return None
    return None


def _build_trace_corn(sim_day: int, context: Dict, decision: float) -> Dict:
    rz = [float(x) for x in context["root_zone_moisture"]]
    lb = [float(x) for x in context["lower_bounds"]]
    margin = [rz[i] - lb[i] for i in range(3)]
    kc = float(context["current_inputs"]["crop_coeff"])
    pet = float(context["current_inputs"]["ref_et_mm_per_day"])
    stage = _build_stage_from_day(sim_day, CORN_PARAMS["stage_days"])
    sensitivity = (
        "CRITICAL"
        if stage == "MID"
        else ("HIGH" if stage == "DEVELOPMENT" else ("MEDIUM" if stage == "LATE" else "LOW"))
    )

    fut_rain = [float(x) for x in context["future_inputs"]["rain_mm_per_day"]]
    fut_et = [float(x) for x in context["future_inputs"]["ref_et_mm_per_day"]]
    fut_kc = [float(x) for x in context["future_inputs"]["crop_coeff"]]
    n_fut = min(len(fut_rain), len(fut_et), len(fut_kc))

    def window(days: int):
        n = min(days, n_fut)
        rain_n = sum(fut_rain[:n])
        etc_n = sum(fut_et[i] * fut_kc[i] for i in range(n))
        return rain_n, etc_n

    rain_3d, etc_3d = window(3)
    sdr_3d = rain_3d / etc_3d if etc_3d > 0.01 else 99.0
    rain_tmr = fut_rain[1] if n_fut > 1 else 0.0
    etc_tmr = fut_et[1] * fut_kc[1] if n_fut > 1 else kc * pet

    hist = context.get("history", {})
    return {
        "day": int(sim_day),
        "allow_irrigation": int(1 if decision >= 0.5 else 0),
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
        "days_since_last_open": int(hist.get("days_since_last_open", 999)),
        "opens_last_7d": int(hist.get("opens_last_7d", 0)),
        "opens_last_14d": int(hist.get("opens_last_14d", 0)),
    }


def call_llm_explainer_corn(prompt: str) -> Optional[str]:
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
                    "model": OLLAMA_MODEL,
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
                timeout = OLLAMA_TIMEOUT_SEC

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
    print(f"[WARN] Corn explainer failed after {LLM_MAX_RETRIES} attempts: {last_error}")
    return None


def _record_explanation_corn(sim_day: int, context: Dict, decision: float, decision_raw_text: Optional[str]) -> None:
    if not ENABLE_EXPLANATION_MODULE or _EXPLANATION_RECORDER is None:
        return
    try:
        trace = _build_trace_corn(sim_day, context, decision)
        explain_prompt = build_global_explanation_prompt(trace)
        explain_raw = call_llm_explainer_corn(explain_prompt)
        explain_obj = parse_global_explanation(explain_raw)
        validation = validate_global_explanation(trace, explain_obj)
        _EXPLANATION_RECORDER.append(
            day=sim_day,
            trace=trace,
            decision_raw_text=decision_raw_text,
            explanation_raw_text=explain_raw,
            parsed_explanation=explain_obj,
            validation=validation,
        )
        _EXPLANATION_RECORDER.write_summary()
    except Exception as e:
        print(f"[WARN] Corn explainability record failed on day {sim_day}: {e}")


def llm_sync_callback_corn(context: Dict) -> Optional[float]:
    global _FALLBACK_WARNED, _SEASON_DAY_COUNTER, _LAST_OPEN_DAY, _OPEN_COUNT_7D, _OPEN_COUNT_14D

    # In once_per_day mode this callback is invoked once each simulation day.
    _SEASON_DAY_COUNTER += 1
    sim_day = _SEASON_DAY_COUNTER

    local_context = dict(context)
    local_context["sim_day"] = sim_day
    local_context["history"] = _history_features(sim_day)

    prompt = build_prompt_exp5_corn(local_context)
    raw_response = call_llm_backend_corn(prompt) if LLM_BACKEND_ENABLED else None
    decision = parse_llm_output_corn(raw_response) if raw_response else None

    if decision is None:
        if STRICT_NO_RULE_FALLBACK:
            raise RuntimeError("LLM decision is invalid; aborting to enforce no-rule-fallback run.")
        if not _FALLBACK_WARNED:
            print("[WARN] Corn EXP5 did not receive a valid LLM decision. Falling back to rule coordinator.")
            _FALLBACK_WARNED = True
        return None

    _record_explanation_corn(sim_day, local_context, decision, raw_response)

    if decision >= 0.5:
        _LAST_OPEN_DAY = sim_day
        _OPEN_COUNT_7D.append(sim_day)
        _OPEN_COUNT_14D.append(sim_day)

    return decision


def _reset_runtime_state() -> None:
    global _FALLBACK_WARNED, _SEASON_DAY_COUNTER, _LAST_OPEN_DAY, _OPEN_COUNT_7D, _OPEN_COUNT_14D
    _FALLBACK_WARNED = False
    _SEASON_DAY_COUNTER = 0
    _LAST_OPEN_DAY = -999
    _OPEN_COUNT_7D = []
    _OPEN_COUNT_14D = []


def main() -> None:
    global _EXPLANATION_RECORDER, _CURRENT_RESULTS_DIR
    apply_corn_overrides(runner, params=CORN_PARAMS)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rel_results_dir = os.path.join(
        "..",
        "zero_retrain_corn_generalization",
        "results",
        f"results_corn_exp5_zero_retrain_{run_id}",
    )
    _CURRENT_RESULTS_DIR = os.path.abspath(os.path.join(BASELINE_DIR, rel_results_dir))
    if ENABLE_EXPLANATION_MODULE:
        _EXPLANATION_RECORDER = ExplanationRecorder(
            results_dir=_CURRENT_RESULTS_DIR,
            experiment_name="Zero-Retrain Corn Generalization - EXP5",
            mode="global_sync",
        )
        atexit.register(_EXPLANATION_RECORDER.write_summary)

    _reset_runtime_state()
    configure_llm(callback=llm_sync_callback_corn, call_policy=LLM_CALL_POLICY)

    print("[INFO] Running zero-retrain corn experiment: EXP5-LLM")
    print(f"[INFO] LLM_PROVIDER={LLM_PROVIDER}")
    if LLM_PROVIDER == "dashscope":
        print(f"[INFO] MODEL={DASHSCOPE_MODEL}")
    else:
        print(f"[INFO] MODEL={OLLAMA_MODEL}")
    print(f"[INFO] results_dir={rel_results_dir}")

    metrics = runner.run_season_long_experiment(
        experiment_name="Zero-Retrain Corn Generalization - EXP5",
        solve_scheduler=SolveScheduler,
        results_dir_name=rel_results_dir,
        simulation_period=123,
    )
    print("[INFO] Finished.")
    print(metrics)


if __name__ == "__main__":
    main()