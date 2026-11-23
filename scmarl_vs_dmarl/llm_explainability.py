import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


GLOBAL_DOMINANT_FACTORS = {
    "stage_risk",
    "soil_deficit",
    "rain_substitution",
    "fragmentation_cost",
    "mixed",
}

EVIDENCE_KEYS = {
    "stage",
    "sensitivity",
    "min_margin",
    "margin_mz1",
    "margin_mz2",
    "margin_mz3",
    "rain_3d",
    "etc_3d",
    "sdr_3d",
    "rain_tomorrow",
    "etc_tomorrow",
    "days_since_last_open",
    "opens_last_7d",
    "opens_last_14d",
}


def _extract_json_object(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()
    if not text:
        return None

    text = text.replace("```json", "```").replace("```JSON", "```")
    if "```" in text:
        parts = [p.strip() for p in text.split("```") if p.strip()]
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                text = part
                break

    m_obj = re.search(r"\{[\s\S]*\}", text)
    candidate = m_obj.group(0) if m_obj else text

    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def build_global_explanation_prompt(trace: Dict[str, Any]) -> str:
    return f"""你是灌溉决策解释器。请解释“已经做出的”当天决策，不要改决策。

已发生决策:
allow_irrigation={int(trace["allow_irrigation"])}

证据:
day={trace["day"]}, stage={trace["stage"]}, sensitivity={trace["sensitivity"]}
margin_mz1={trace["margin_mz1"]:+.4f}, margin_mz2={trace["margin_mz2"]:+.4f}, margin_mz3={trace["margin_mz3"]:+.4f}, min_margin={trace["min_margin"]:+.4f}
rain_3d={trace["rain_3d"]:.2f}, etc_3d={trace["etc_3d"]:.2f}, sdr_3d={trace["sdr_3d"]:.2f}
rain_tomorrow={trace["rain_tomorrow"]:.2f}, etc_tomorrow={trace["etc_tomorrow"]:.2f}
days_since_last_open={trace.get("days_since_last_open", 999)}, opens_last_7d={trace.get("opens_last_7d", 0)}, opens_last_14d={trace.get("opens_last_14d", 0)}

要求:
1) 解释必须与allow_irrigation一致。
2) 必须引用至少2个证据变量。
3) counterfactual写“如果明天雨量变化，会不会改判”。

只输出一行JSON:
{{
  "allow_irrigation": 0 or 1,
  "dominant_factor": "stage_risk|soil_deficit|rain_substitution|fragmentation_cost|mixed",
  "agronomic_reason": "一句话",
  "weather_reason": "一句话",
  "cost_reason": "一句话",
  "counterfactual": "一句话",
  "evidence_keys": ["min_margin","sdr_3d"],
  "confidence": 0.00
}}"""


def build_per_zone_explanation_prompt(trace: Dict[str, Any]) -> str:
    return f"""你是灌溉决策解释器。请解释“已经做出的”分区决策，不要改决策。

已发生决策:
allow_mz1={int(trace["allow_mz1"])}, allow_mz2={int(trace["allow_mz2"])}, allow_mz3={int(trace["allow_mz3"])}

证据:
day={trace["day"]}, stage={trace["stage"]}, sensitivity={trace["sensitivity"]}
margin_mz1={trace["margin_mz1"]:+.4f}, margin_mz2={trace["margin_mz2"]:+.4f}, margin_mz3={trace["margin_mz3"]:+.4f}, min_margin={trace["min_margin"]:+.4f}
rain_3d={trace["rain_3d"]:.2f}, etc_3d={trace["etc_3d"]:.2f}, sdr_3d={trace["sdr_3d"]:.2f}
rain_tomorrow={trace["rain_tomorrow"]:.2f}, etc_tomorrow={trace["etc_tomorrow"]:.2f}

要求:
1) 解释必须与3个分区决策一致。
2) 必须指出主导分区（若都不开则NONE）。
3) 必须引用至少2个证据变量。

只输出一行JSON:
{{
  "allow_mz1": 0 or 1,
  "allow_mz2": 0 or 1,
  "allow_mz3": 0 or 1,
  "dominant_zone": "MZ1|MZ2|MZ3|NONE",
  "dominant_factor": "stage_risk|soil_deficit|rain_substitution|fragmentation_cost|mixed",
  "reason": "一句话",
  "counterfactual": "一句话",
  "evidence_keys": ["margin_mz1","sdr_3d"],
  "confidence": 0.00
}}"""


def parse_global_explanation(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None
    required = [
        "allow_irrigation",
        "dominant_factor",
        "agronomic_reason",
        "weather_reason",
        "cost_reason",
        "counterfactual",
        "evidence_keys",
        "confidence",
    ]
    if any(k not in obj for k in required):
        return None
    return obj


def parse_per_zone_explanation(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None
    required = [
        "allow_mz1",
        "allow_mz2",
        "allow_mz3",
        "dominant_zone",
        "dominant_factor",
        "reason",
        "counterfactual",
        "evidence_keys",
        "confidence",
    ]
    if any(k not in obj for k in required):
        return None
    return obj


def _safe_bool_like01(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "allow", "open"):
            return 1
        if v in ("0", "false", "no", "deny", "close"):
            return 0
        try:
            return 1 if float(v) >= 0.5 else 0
        except Exception:
            return None
    return None


def _safe_float_01(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if x < 0.0 or x > 1.0:
        return None
    return x


def validate_global_explanation(trace: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if parsed is None:
        return {
            "parse_ok": False,
            "decision_match": False,
            "fields_ok": False,
            "evidence_keys_ok": False,
            "confidence_ok": False,
            "data_sanity_ok": False,
            "overall_ok": False,
        }

    decision = _safe_bool_like01(parsed.get("allow_irrigation"))
    decision_match = decision is not None and decision == int(trace["allow_irrigation"])
    fields_ok = isinstance(parsed.get("dominant_factor"), str) and parsed.get("dominant_factor") in GLOBAL_DOMINANT_FACTORS

    ev_keys = parsed.get("evidence_keys")
    evidence_keys_ok = isinstance(ev_keys, list) and len(ev_keys) >= 2 and all(
        isinstance(k, str) and k in EVIDENCE_KEYS for k in ev_keys
    )
    confidence_ok = _safe_float_01(parsed.get("confidence")) is not None

    min_margin = float(trace["min_margin"])
    sdr_3d = float(trace["sdr_3d"])
    rain_tmr = float(trace["rain_tomorrow"])
    etc_tmr = float(trace["etc_tomorrow"])
    is_sensitive = str(trace["sensitivity"]) in ("HIGH", "CRITICAL")

    if decision is None:
        data_sanity_ok = False
    elif decision == 1:
        data_sanity_ok = not (min_margin >= 0.015 and sdr_3d >= 1.0 and rain_tmr >= 0.8 * etc_tmr)
    else:
        data_sanity_ok = not (min_margin <= -0.003 and sdr_3d < 0.70 and is_sensitive)

    overall_ok = all([decision_match, fields_ok, evidence_keys_ok, confidence_ok, data_sanity_ok])
    return {
        "parse_ok": True,
        "decision_match": bool(decision_match),
        "fields_ok": bool(fields_ok),
        "evidence_keys_ok": bool(evidence_keys_ok),
        "confidence_ok": bool(confidence_ok),
        "data_sanity_ok": bool(data_sanity_ok),
        "overall_ok": bool(overall_ok),
    }


def validate_per_zone_explanation(trace: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if parsed is None:
        return {
            "parse_ok": False,
            "decision_match": False,
            "fields_ok": False,
            "evidence_keys_ok": False,
            "confidence_ok": False,
            "data_sanity_ok": False,
            "overall_ok": False,
        }

    d1 = _safe_bool_like01(parsed.get("allow_mz1"))
    d2 = _safe_bool_like01(parsed.get("allow_mz2"))
    d3 = _safe_bool_like01(parsed.get("allow_mz3"))
    decision_match = (
        d1 is not None
        and d2 is not None
        and d3 is not None
        and d1 == int(trace["allow_mz1"])
        and d2 == int(trace["allow_mz2"])
        and d3 == int(trace["allow_mz3"])
    )

    dominant_zone = parsed.get("dominant_zone")
    fields_ok = (
        isinstance(dominant_zone, str)
        and dominant_zone in ("MZ1", "MZ2", "MZ3", "NONE")
        and isinstance(parsed.get("dominant_factor"), str)
        and parsed.get("dominant_factor") in GLOBAL_DOMINANT_FACTORS
    )

    ev_keys = parsed.get("evidence_keys")
    evidence_keys_ok = isinstance(ev_keys, list) and len(ev_keys) >= 2 and all(
        isinstance(k, str) and k in EVIDENCE_KEYS for k in ev_keys
    )
    confidence_ok = _safe_float_01(parsed.get("confidence")) is not None

    data_sanity_ok = True
    overall_ok = all([decision_match, fields_ok, evidence_keys_ok, confidence_ok, data_sanity_ok])
    return {
        "parse_ok": True,
        "decision_match": bool(decision_match),
        "fields_ok": bool(fields_ok),
        "evidence_keys_ok": bool(evidence_keys_ok),
        "confidence_ok": bool(confidence_ok),
        "data_sanity_ok": bool(data_sanity_ok),
        "overall_ok": bool(overall_ok),
    }


class ExplanationRecorder:
    def __init__(self, results_dir: str, experiment_name: str, mode: str) -> None:
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.mode = mode
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.records_path = os.path.join(results_dir, "llm_explanations.jsonl")
        self.summary_path = os.path.join(results_dir, "llm_explainability_summary.json")
        self.stats = {
            "total": 0,
            "parse_ok": 0,
            "decision_match": 0,
            "fields_ok": 0,
            "evidence_keys_ok": 0,
            "confidence_ok": 0,
            "data_sanity_ok": 0,
            "overall_ok": 0,
        }
        os.makedirs(results_dir, exist_ok=True)

    def append(
        self,
        day: int,
        trace: Dict[str, Any],
        decision_raw_text: Optional[str],
        explanation_raw_text: Optional[str],
        parsed_explanation: Optional[Dict[str, Any]],
        validation: Dict[str, Any],
    ) -> None:
        self.stats["total"] += 1
        for key in ("parse_ok", "decision_match", "fields_ok", "evidence_keys_ok", "confidence_ok", "data_sanity_ok", "overall_ok"):
            if key in self.stats and bool(validation.get(key, False)):
                self.stats[key] += 1

        record = {
            "day": int(day),
            "mode": self.mode,
            "trace": trace,
            "decision_raw_text": decision_raw_text,
            "explanation_raw_text": explanation_raw_text,
            "parsed_explanation": parsed_explanation,
            "validation": validation,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self) -> None:
        total = max(self.stats["total"], 1)
        summary = {
            "experiment_name": self.experiment_name,
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "counts": self.stats,
            "rates": {
                "parse_ok_rate": self.stats["parse_ok"] / total,
                "decision_match_rate": self.stats["decision_match"] / total,
                "fields_ok_rate": self.stats["fields_ok"] / total,
                "evidence_keys_ok_rate": self.stats["evidence_keys_ok"] / total,
                "confidence_ok_rate": self.stats["confidence_ok"] / total,
                "data_sanity_ok_rate": self.stats["data_sanity_ok"] / total if "data_sanity_ok" in self.stats else None,
                "overall_ok_rate": self.stats["overall_ok"] / total,
            },
            "records_file": self.records_path,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
