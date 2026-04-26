import numpy as np

def compute_indicators(results: dict, seg_map: np.ndarray) -> dict:
    """
    Rule-based accessibility scoring from placement results.
    Returns scores and reasons for each indicator.
    """
    indicators = {
        "connectivity":  {"score": 0, "max": 1, "reason": "No crosswalk placed"},
        "comfort":       {"score": 0, "max": 1, "reason": "No bench placed"},
        "mobility":      {"score": 0, "max": 1, "reason": "No curb ramp placed"},
        "overall":       {"score": 0, "max": 3}
    }

    if results.get("crosswalk", {}).get("valid"):
        indicators["connectivity"] = {
            "score": 1, "max": 1,
            "reason": "Crosswalk connects road to sidewalk"
        }

    if results.get("bench", {}).get("valid"):
        indicators["comfort"] = {
            "score": 1, "max": 1,
            "reason": "Bench placed on clear sidewalk"
        }

    if results.get("curb_ramp", {}).get("valid"):
        indicators["mobility"] = {
            "score": 1, "max": 1,
            "reason": "Curb ramp at road-sidewalk boundary"
        }

    indicators["overall"]["score"] = (
        indicators["connectivity"]["score"] +
        indicators["comfort"]["score"] +
        indicators["mobility"]["score"]
    )
    return indicators


def format_indicators(indicators: dict) -> str:
    lines = ["ACCESSIBILITY INDICATORS", "=" * 30]
    for key in ["connectivity", "comfort", "mobility"]:
        ind = indicators[key]
        bar = "█" * ind["score"] + "░" * (ind["max"] - ind["score"])
        lines.append(f"{key.upper():15s} [{bar}] {ind['score']}/{ind['max']}")
        lines.append(f"  {ind['reason']}")
    lines.append("=" * 30)
    lines.append(f"OVERALL: {indicators['overall']['score']}/3")
    return "\n".join(lines)