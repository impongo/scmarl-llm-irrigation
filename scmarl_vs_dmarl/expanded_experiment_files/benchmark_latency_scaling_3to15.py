import argparse
import contextlib
import io
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, "core_scmarl_vs_dmarl")
COMMON_ACTUAL_FIELD = os.path.abspath(os.path.join(BASE_DIR, "..", "common", "actual_field"))
COMMON_WEATHER = os.path.abspath(os.path.join(BASE_DIR, "..", "common", "weather_data"))
COMMON_INITIAL = os.path.abspath(os.path.join(BASE_DIR, "..", "common", "initial_states"))
COMMON_ENVIRONMENTS = os.path.abspath(os.path.join(BASE_DIR, "..", "common", "environments"))

# Legacy scheduler modules load relative files at import time.
# Ensure their expected working directory is active before importing.
os.chdir(BASE_DIR)


import sys  # noqa: E402

sys.path.append(CORE_DIR)
sys.path.append(COMMON_ACTUAL_FIELD)
sys.path.append(COMMON_ENVIRONMENTS)

from crop_coefficient_calculation import generate_Kc  # noqa: E402
from scheduler_evaluation_dmarl import SolveScheduler as solve_dmarl  # noqa: E402
from scheduler_evaluation_exp5_llm_sync_scmarl_local import (  # noqa: E402
    SolveScheduler as solve_exp5,
    configure_llm as configure_exp5_llm,
)
from scheduler_evaluation_scmarl import SolveScheduler as solve_scmarl  # noqa: E402
from exp5_llm_sync_season_long import (  # noqa: E402
    build_prompt_exp5,
    call_llm_backend_exp5,
    parse_llm_output_exp5,
)


@dataclass
class BenchmarkInput:
    state_mz1: np.ndarray
    state_mz2: np.ndarray
    state_mz3: np.ndarray
    et_seq: np.ndarray
    kc_seq: np.ndarray
    lai_seq: np.ndarray
    rain_seq: np.ndarray


def _load_vector(path: str) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path)).astype(float).reshape(-1)


def _load_benchmark_input(day_index: int = 0, horizon: int = 14) -> BenchmarkInput:
    state_mz1 = _load_vector(os.path.join(COMMON_INITIAL, "Init_state_all_mz1_vrd_lai.txt"))
    state_mz2 = _load_vector(os.path.join(COMMON_INITIAL, "Init_state_all_mz2_vrd_lai.txt"))
    state_mz3 = _load_vector(os.path.join(COMMON_INITIAL, "Init_state_all_mz3_vrd_lai.txt"))

    avg_temp = _load_vector(os.path.join(COMMON_WEATHER, "dAVG_TEMP.txt"))
    kc_all = np.atleast_1d(generate_Kc(avg_temp)).astype(float).reshape(-1)
    et_mm_day = np.clip(_load_vector(os.path.join(COMMON_WEATHER, "dPET.txt")), 1.04, 9.0)
    rain_mm_day = _load_vector(os.path.join(COMMON_WEATHER, "dRAIN.txt"))

    et_seq = et_mm_day[day_index : day_index + horizon] / (86400.0 * 1000.0)
    kc_seq = kc_all[day_index : day_index + horizon]
    lai_seq = np.ones_like(et_seq, dtype=float)
    rain_seq = -rain_mm_day[day_index : day_index + horizon] / 1000.0

    if min(len(et_seq), len(kc_seq), len(lai_seq), len(rain_seq)) < horizon:
        raise RuntimeError("Weather data shorter than requested benchmark horizon.")

    return BenchmarkInput(
        state_mz1=1.05 * state_mz1,
        state_mz2=1.05 * state_mz2,
        state_mz3=1.08 * state_mz3,
        et_seq=et_seq,
        kc_seq=kc_seq,
        lai_seq=lai_seq,
        rain_seq=rain_seq,
    )


def _mock_llm_sync_callback(context: Dict) -> float:
    rz = np.asarray(context["root_zone_moisture"], dtype=float)
    lb = np.asarray(context["lower_bounds"], dtype=float)
    margin = rz - lb

    kc = float(context["current_inputs"]["crop_coeff"])
    fut_rain = np.asarray(context["future_inputs"]["rain_mm_per_day"], dtype=float)
    fut_et = np.asarray(context["future_inputs"]["ref_et_mm_per_day"], dtype=float)
    fut_kc = np.asarray(context["future_inputs"]["crop_coeff"], dtype=float)
    n3 = int(min(3, len(fut_rain), len(fut_et), len(fut_kc)))
    if n3 > 0:
        rain3 = float(np.sum(fut_rain[:n3]))
        etc3 = float(np.sum(fut_et[:n3] * fut_kc[:n3]))
    else:
        rain3 = 0.0
        etc3 = max(0.1, kc * float(context["current_inputs"]["ref_et_mm_per_day"]) * 3.0)

    sensitive = kc >= 0.85

    if np.any(margin < 0.0):
        return 1.0
    if sensitive and np.min(margin) < 0.006 and rain3 < 0.70 * etc3:
        return 1.0
    if np.min(margin) >= 0.010 and rain3 >= 0.80 * etc3:
        return 0.0
    return 0.0


def _api_llm_sync_callback(context: Dict) -> float:
    prompt = build_prompt_exp5(context)
    raw = call_llm_backend_exp5(prompt)
    decision = parse_llm_output_exp5(raw) if raw else None
    if decision is None:
        raise RuntimeError("EXP5 API callback returned invalid decision.")
    return float(decision)


def _build_group_state(base_state: np.ndarray, group_index: int, scale: float) -> np.ndarray:
    # Small deterministic perturbation so all groups are not perfectly identical.
    # Magnitude kept tiny to stay near the baseline physical operating range.
    delta = (group_index % 5) * 0.0005 * scale
    return base_state + delta


def _silent_solve(
    solve_fn: Callable,
    state_mz1: np.ndarray,
    state_mz2: np.ndarray,
    state_mz3: np.ndarray,
    et_seq: np.ndarray,
    kc_seq: np.ndarray,
    lai_seq: np.ndarray,
    rain_seq: np.ndarray,
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        solve_fn(
            state_mz1,
            state_mz2,
            state_mz3,
            et_seq,
            kc_seq,
            lai_seq,
            rain_seq,
        )


def _run_one_day_latency(
    solve_fn: Callable,
    bench_input: BenchmarkInput,
    zone_count: int,
) -> float:
    groups = zone_count // 3
    start = time.perf_counter()
    for group_idx in range(groups):
        state_mz1 = _build_group_state(bench_input.state_mz1, group_idx, scale=1.0)
        state_mz2 = _build_group_state(bench_input.state_mz2, group_idx, scale=1.1)
        state_mz3 = _build_group_state(bench_input.state_mz3, group_idx, scale=1.2)
        _silent_solve(
            solve_fn,
            state_mz1,
            state_mz2,
            state_mz3,
            bench_input.et_seq,
            bench_input.kc_seq,
            bench_input.lai_seq,
            bench_input.rain_seq,
        )
    end = time.perf_counter()
    return end - start


def _mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(sum(arr) / len(arr)) if arr else float("nan")


def _p90(values: Iterable[float]) -> float:
    arr = sorted(values)
    if not arr:
        return float("nan")
    if len(arr) == 1:
        return float(arr[0])
    return float(arr[int(round(0.9 * (len(arr) - 1)))])


def run_benchmark(zone_counts: List[int], repeats: int, warmup: bool, llm_mode: str) -> List[Dict]:
    bench_input = _load_benchmark_input(day_index=0, horizon=14)

    if llm_mode == "api":
        configure_exp5_llm(callback=_api_llm_sync_callback, call_policy="once_per_day")
    else:
        configure_exp5_llm(callback=_mock_llm_sync_callback, call_policy="once_per_day")

    methods: List[Tuple[str, Callable]] = [
        ("SCMARL", solve_scmarl),
        ("DMARL", solve_dmarl),
        ("EXP5", solve_exp5),
    ]

    results: List[Dict] = []
    for method_name, solver in methods:
        print(f"\n[INFO] Benchmarking {method_name} ...")
        for zone_count in zone_counts:
            if zone_count % 3 != 0:
                raise ValueError(f"zone_count must be multiple of 3, got {zone_count}")

            if warmup:
                _run_one_day_latency(solver, bench_input, zone_count=3)

            samples: List[float] = []
            for rep in range(repeats):
                latency_sec = _run_one_day_latency(solver, bench_input, zone_count=zone_count)
                samples.append(latency_sec)
                print(
                    f"[DEBUG] {method_name} zones={zone_count} repeat={rep + 1}/{repeats} "
                    f"latency={latency_sec:.3f}s"
                )

            avg_sec = _mean(samples)
            p90_sec = _p90(samples)
            avg_ms_per_zone = (avg_sec * 1000.0 / zone_count) if zone_count > 0 else float("nan")
            results.append(
                {
                    "method": method_name,
                    "zones": zone_count,
                    "groups": zone_count // 3,
                    "avg_sec": avg_sec,
                    "p90_sec": p90_sec,
                    "std_sec": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
                    "avg_ms_per_zone": avg_ms_per_zone,
                }
            )
    return results


def print_table(results: List[Dict]) -> None:
    print("\n================ Latency Scaling Results ================")
    print("Method | Zones | Groups(3-zone blocks) | Avg Latency(s/day) | P90(s) | Std(s) | Avg(ms/zone)")
    print("-" * 96)
    for row in results:
        print(
            f"{row['method']:>6} | "
            f"{row['zones']:>5} | "
            f"{row['groups']:>20} | "
            f"{row['avg_sec']:>17.3f} | "
            f"{row['p90_sec']:>6.3f} | "
            f"{row['std_sec']:>6.3f} | "
            f"{row['avg_ms_per_zone']:>12.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency scaling benchmark for SCMARL/DMARL/EXP5 without CL.")
    parser.add_argument("--zone-counts", type=int, nargs="+", default=[3, 6, 9, 12, 15])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--llm-mode", choices=["mock", "api"], default="mock")
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    print(f"[INFO] Working directory: {BASE_DIR}")
    print(f"[INFO] Zone counts: {args.zone_counts}, repeats={args.repeats}, warmup={not args.no_warmup}")
    if args.llm_mode == "api":
        print("[INFO] EXP5 uses REAL API callback (network/API latency included).")
    else:
        print("[INFO] EXP5 uses mock LLM callback for stable local latency measurement (no external API delay).")

    results = run_benchmark(
        zone_counts=list(args.zone_counts),
        repeats=int(args.repeats),
        warmup=(not args.no_warmup),
        llm_mode=args.llm_mode,
    )
    print_table(results)


if __name__ == "__main__":
    main()
