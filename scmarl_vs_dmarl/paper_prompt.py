from typing import Iterable, Mapping

DECISION_SYSTEM_MESSAGE = (
    "You are an agricultural-irrigation coordinator. Judge the gate by "
    "combining crop-stage knowledge, soil status, and weather forecasts. "
    "Thresholds are decision aids rather than rules to be executed mechanically. "
    "Opening has a fixed cost, so avoid fragmented openings. The local agent "
    "determines the irrigation amount. Return JSON only."
)

EXPLANATION_SYSTEM_MESSAGE = (
    "You are an irrigation-decision explainer. Explain only the decision that "
    "has already been executed; do not change it or propose another action. "
    "Return valid JSON only, without Markdown."
)


class GateHistory:
    def __init__(self):
        self.open_days = []

    def features(self, day: int) -> dict:
        previous = [d for d in self.open_days if d < day]
        return {
            "days_since_last_open": day - previous[-1] if previous else 999,
            "opens_last_7d": sum(0 < day - d <= 7 for d in previous),
        }

    def record(self, day: int, opened: bool) -> None:
        if opened and (not self.open_days or self.open_days[-1] != day):
            self.open_days.append(day)

    def reset(self) -> None:
        self.open_days.clear()


def build_paper_decision_prompt(*, crop: str, day: int, stage: str,
        sensitivity: str, kc: float, pet: float, etc_today: float, lai: float,
        margins: Iterable[float], sdr_3d: float, rain_tomorrow: float,
        etc_tomorrow: float, history: Mapping[str, int], per_zone: bool = False) -> str:
    margins = list(margins)
    zone_count = len(margins)
    margin_lines = "\n".join(f"MZ{i + 1} = {value:+.4f}" for i, value in enumerate(margins))
    min_margin = min(margins)
    worst_zone = margins.index(min_margin) + 1
    field_label = f"a {zone_count}-zone crop field"
    if per_zone:
        decision_line = "Make one decision today: allow_mz1 ... allow_mzN (each 0 or 1)."
        local_line = "If a zone gate is 1, its local agent determines that zone's irrigation amount."
        output = "{" + ", ".join(f'\"allow_mz{i + 1}\": 0 or 1' for i in range(zone_count)) + "}"
    else:
        decision_line = "Make one decision today: allow_irrigation = 0 or 1."
        local_line = "If allow_irrigation = 1, the local agents determine the irrigation amounts."
        output = ("{\"allow_irrigation\": 0 or 1, \"stress_risk\": \"LOW|MEDIUM|HIGH|CRITICAL\", "
                  "\"rain_substitution_risk\": \"LOW|MEDIUM|HIGH\", \"fragmentation_risk\": \"LOW|MEDIUM|HIGH\", "
                  "\"confidence\": 0.00, \"reason\": \"one concise sentence integrating crop stage, soil, weather, and opening-frequency cost\"}")
    return f"""You are the irrigation coordinator for {field_label} of {crop}.
{decision_line}

You are not a numerical optimizer and do not calculate irrigation amounts. {local_line}

Objectives, in priority order:
1. Avoid yield loss from water stress in HIGH/CRITICAL stages.
2. Improve irrigation water use efficiency (IWUE).
3. Avoid frequent fragmented gate openings, which have a fixed cost.

TODAY INPUTS
day = {day}
stage = {stage}
sensitivity = {sensitivity}
Kc = {kc:.3f}
PET = {pet:.2f} mm/day
ETc_today = {etc_today:.2f} mm/day
LAI = {lai:.2f}

Soil margins (root-zone moisture minus lower stress bound):
{margin_lines}
min_margin = {min_margin:+.4f}; worst_zone = {worst_zone}

Weather-demand information:
sdr_3d = {sdr_3d:.2f}
rain_tomorrow = {rain_tomorrow:.2f} mm
ETc_tomorrow = {etc_tomorrow:.2f} mm

History:
- days_since_last_open: {history.get('days_since_last_open', 999)}
- opens_last_7d: {history.get('opens_last_7d', 0)}

DECISION PRINCIPLES
A. In HIGH/CRITICAL stages, protect against deep water deficit when the near-term forecast is dry. In LOW/MEDIUM stages, be more conservative and prefer rainfall substitution.
B. Lower min_margin means greater stress risk. If min_margin is positive and short-term rainfall can substitute for irrigation, do not irrigate pre-emptively.
C. If rainfall in the next 1--3 days can cover near-term ETc, delay irrigation unless the risk that the rainfall arrives too late is high.
D. Gate opening has a fixed cost, so use a conservative default. The objective is to reduce the number and cost of openings and avoid frequent small irrigation events.

THRESHOLD AIDS
- Evidence for opening should be jointly strong; for example, sensitivity in {{HIGH, CRITICAL}} and min_margin < 0.
- If min_margin >= 0 and sensitivity in {{LOW, MEDIUM}}, default to 0.
- If evidence is mixed or insufficient, default to 0 because opening has a fixed cost.

OUTPUT FORMAT: return exactly one JSON object on one line, without Markdown or additional text:
{output}"""