import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import numpy as np

from core_scmarl_vs_dmarl.scheduler_evaluation_exp5_llm_sync_scmarl_local import (
    SolveScheduler,
    configure_llm,
)
from season_long_runner_exp456 import run_season_long_experiment


PAPER_PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PAPER_PROMPT_DIR not in sys.path:
    sys.path.insert(0, PAPER_PROMPT_DIR)
from paper_prompt import DECISION_SYSTEM_MESSAGE, EXPLANATION_SYSTEM_MESSAGE, GateHistory, build_paper_decision_prompt
_PAPER_GATE_HISTORY = GateHistory()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_NAME = "Experiment 5 Expanded - SCMARL-LLM Sync (Global 0/1) - SCMARL local agents"
RESULTS_DIR_NAME = "results_exp5_expanded"

LLM_BACKEND_ENABLED = True
_FALLBACK_WARNED = False

LLM_CALL_POLICY = "once_per_day"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "qwen-plus"
LLM_TIMEOUT_SEC = 90

LLM_PROVIDER = "dashscope"  # "dashscope" or "ollama"
DASHSCOPE_URL = os.getenv("DASHSCOPE_API_URL", "").strip()
DASHSCOPE_MODEL = "qwen3.5-27b"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_TIMEOUT_SEC = 240

LLM_MAX_RETRIES = 4
LLM_RETRY_SLEEP_SEC = 2.0
STRICT_NO_RULE_FALLBACK = True


def build_prompt_exp5(context: Dict) -> str:
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
        etc_tomorrow=etc_tomorrow, history=history, per_zone=False)
def call_llm_backend_exp5(prompt: str) -> Optional[str]:
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
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9,
                        "num_predict": 80,
                    },
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
            else:
                text = obj.get("response", "")
                if not text:
                    text = obj.get("thinking", "")
                if not text and isinstance(obj.get("message"), dict):
                    text = obj["message"].get("content") or obj["message"].get("thinking") or ""

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


def parse_llm_output_exp5(raw_text: str) -> Optional[float]:
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

    data = None
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"allow[_\s-]*irrigation[^01tf]*(0|1|true|false)", text, re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            if v in ("1", "true"):
                return 1.0
            if v in ("0", "false"):
                return 0.0
        return None

    if not isinstance(data, dict):
        return None

    for key in ("allow_irrigation", "allow", "decision", "global_allow"):
        if key in data:
            value = data[key]
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


def llm_sync_callback(context: Dict) -> Optional[float]:
    global _FALLBACK_WARNED
    prompt = build_prompt_exp5(context)
    raw_response = call_llm_backend_exp5(prompt) if LLM_BACKEND_ENABLED else None
    decision = parse_llm_output_exp5(raw_response) if raw_response else None
    if decision is not None:
        day = int(context.get("sim_day", context.get("step_index", 0) + 1))
        _PAPER_GATE_HISTORY.record(day, float(decision) >= 0.5)

    if decision is None:
        if STRICT_NO_RULE_FALLBACK:
            raise RuntimeError("LLM decision is invalid; aborting to enforce no-rule-fallback run.")
        if not _FALLBACK_WARNED:
            print("[WARN] Experiment 5 did not receive a valid LLM decision. Falling back to rule coordinator.")
            _FALLBACK_WARNED = True
    return decision


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _aggregate_group_metrics(group_metrics: List[Dict]) -> Dict:
    zone_yields: List[float] = []
    zone_irrig: List[float] = []
    for gm in group_metrics:
        zone_yields.extend([float(gm["yield_mz1"]), float(gm["yield_mz2"]), float(gm["yield_mz3"])])
        zone_irrig.extend([float(gm["irrigation_mz1"]), float(gm["irrigation_mz2"]), float(gm["irrigation_mz3"])])

    zone_yields_arr = np.asarray(zone_yields, dtype=float)
    zone_irrig_arr = np.asarray(zone_irrig, dtype=float)

    avg_yield = float(np.mean(zone_yields_arr))
    avg_irrig = float(np.mean(zone_irrig_arr))
    total_irrig = float(np.sum(zone_irrig_arr))
    total_yield = float(np.sum(zone_yields_arr))

    iwue_physical = float(total_yield / total_irrig) if total_irrig > 1e-12 else float("nan")
    iwue_paper_style = float(avg_yield / total_irrig) if total_irrig > 1e-12 else float("nan")

    per_zone_iwue = np.divide(
        zone_yields_arr,
        zone_irrig_arr,
        out=np.full_like(zone_yields_arr, np.nan),
        where=zone_irrig_arr > 1e-12,
    )

    return {
        "zone_count": int(len(zone_yields)),
        "avg_yield_kg_m2": avg_yield,
        "total_yield_sum_over_zones_kg_m2": total_yield,
        "avg_irrigation_m3_m2": avg_irrig,
        "total_irrigation_m3_m2": total_irrig,
        "iwue_physical_kg_m3": iwue_physical,
        "iwue_paper_style_kg_m3": iwue_paper_style,
        "per_zone_iwue_mean_kg_m3": float(np.nanmean(per_zone_iwue)),
        "per_zone_iwue_std_kg_m3": float(np.nanstd(per_zone_iwue)),
    }


def run_expanded_exp5(zone_counts: List[int], simulation_period: int = 123) -> List[Dict]:
    run_id = _timestamp()
    out_root = os.path.join(BASE_DIR, f"{RESULTS_DIR_NAME}_{run_id}")
    _ensure_dir(out_root)

    all_rows: List[Dict] = []
    print(f"[INFO] Start expanded EXP5 experiment: run_id={run_id}")
    print(f"[INFO] Output root: {out_root}")

    configure_llm(callback=llm_sync_callback, call_policy=LLM_CALL_POLICY)

    for zone_count in zone_counts:
        if zone_count % 3 != 0:
            raise ValueError(f"zone_count must be multiple of 3, got {zone_count}")

        groups = zone_count // 3
        print(f"\n[INFO] Running EXP5 for {zone_count} plots ({groups} groups x 3 plots)...")
        group_metrics: List[Dict] = []

        for g in range(groups):
            rel_results_dir = os.path.relpath(
                os.path.join(out_root, f"z{zone_count}_group{g + 1}"),
                BASE_DIR,
            )
            print(f"[INFO] -> Group {g + 1}/{groups}, results_dir={rel_results_dir}")
            _PAPER_GATE_HISTORY.reset()
            metrics = run_season_long_experiment(
                experiment_name=f"{EXPERIMENT_NAME} z{zone_count} group{g + 1}",
                solve_scheduler=SolveScheduler,
                results_dir_name=rel_results_dir,
                simulation_period=simulation_period,
            )
            group_metrics.append(metrics)

        agg = _aggregate_group_metrics(group_metrics)
        agg["method"] = "EXP5"
        agg["groups"] = groups
        agg["run_id"] = run_id
        all_rows.append(agg)

        print("[INFO] Aggregated:")
        print(
            f"  avg_yield={agg['avg_yield_kg_m2']:.4f} kg/m^2, "
            f"total_irrig={agg['total_irrigation_m3_m2']:.4f} m^3/m^2, "
            f"iwue_physical={agg['iwue_physical_kg_m3']:.4f} kg/m^3"
        )

    csv_path = os.path.join(out_root, "summary_exp5_expanded.csv")
    json_path = os.path.join(out_root, "summary_exp5_expanded.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "run_id",
                "zone_count",
                "groups",
                "avg_yield_kg_m2",
                "total_yield_sum_over_zones_kg_m2",
                "avg_irrigation_m3_m2",
                "total_irrigation_m3_m2",
                "iwue_physical_kg_m3",
                "iwue_paper_style_kg_m3",
                "per_zone_iwue_mean_kg_m3",
                "per_zone_iwue_std_kg_m3",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] CSV summary: {csv_path}")
    print(f"[INFO] JSON summary: {json_path}")
    return all_rows


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    run_expanded_exp5(zone_counts=[9, 15], simulation_period=123)
