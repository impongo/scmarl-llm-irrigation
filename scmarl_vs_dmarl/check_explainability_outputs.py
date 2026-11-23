import json
import os
from typing import Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(name: str, summary_path: str) -> None:
    if not os.path.exists(summary_path):
        print(f"[MISS] {name}: {summary_path}")
        return

    obj = _load_json(summary_path)
    rates = obj.get("rates", {})
    counts = obj.get("counts", {})

    print(f"\n=== {name} ===")
    print(f"summary: {summary_path}")
    print(f"total days logged: {counts.get('total', 0)}")
    print(f"parse_ok_rate: {rates.get('parse_ok_rate', 0):.3f}")
    print(f"decision_match_rate: {rates.get('decision_match_rate', 0):.3f}")
    print(f"fields_ok_rate: {rates.get('fields_ok_rate', 0):.3f}")
    print(f"evidence_keys_ok_rate: {rates.get('evidence_keys_ok_rate', 0):.3f}")
    print(f"confidence_ok_rate: {rates.get('confidence_ok_rate', 0):.3f}")
    ds = rates.get("data_sanity_ok_rate", None)
    if ds is not None:
        print(f"data_sanity_ok_rate: {ds:.3f}")
    print(f"overall_ok_rate: {rates.get('overall_ok_rate', 0):.3f}")
    print(f"records_file: {obj.get('records_file', '')}")


def main() -> None:
    exp5_summary = os.path.join(BASE_DIR, "results_exp5_llm_sync_scmarl_local", "llm_explainability_summary.json")
    exp6_summary = os.path.join(BASE_DIR, "results_exp6_llm_per_zone_scmarl_local", "llm_explainability_summary.json")

    _print_summary("EXP5 explainability", exp5_summary)
    _print_summary("EXP6 explainability", exp6_summary)


if __name__ == "__main__":
    main()

