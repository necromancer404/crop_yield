"""Rule-based fertilizer and yield improvement suggestions."""

from __future__ import annotations

from models.schemas import CropPredictRequest


def fertilizer_suggestions(req: CropPredictRequest) -> list[str]:
    out: list[str] = []
    if req.N < 40:
        out.append(
            "Nitrogen is low: consider urea or organic manure, and legume rotation to build soil N."
        )
    elif req.N > 120:
        out.append("Nitrogen is high: reduce N fertilizer to avoid lodging and runoff.")

    if req.P < 35:
        out.append("Phosphorus is low: apply phosphate-rich fertilizer or bone meal where appropriate.")
    elif req.P > 100:
        out.append("Phosphorus is elevated: verify soil test before adding more P.")

    if req.K < 35:
        out.append("Potassium is low: use potash or wood ash (where suitable) to improve stress tolerance.")
    elif req.K > 100:
        out.append("Potassium is high: avoid further K until retested.")

    if req.ph < 6.0:
        out.append("pH is acidic: liming may improve nutrient availability (confirm with soil test).")
    elif req.ph > 8.0:
        out.append("pH is alkaline: organic matter and targeted amendments can improve micronutrient uptake.")

    if req.rainfall < 80:
        out.append("Rainfall is low: prioritize drought-tolerant varieties, mulching, and efficient irrigation.")
    elif req.rainfall > 250:
        out.append("High rainfall: ensure drainage to reduce waterlogging and nutrient leaching.")

    if not out:
        out.append("Soil macronutrients look balanced; maintain with regular soil testing and crop rotation.")
    return out


def yield_insights(predicted_yield: float, yield_for_recommended: float | None) -> list[str]:
    tips: list[str] = []
    if predicted_yield <= 0:
        tips.append("Predicted yield is non-positive; check inputs and units (area > 0).")
        return tips

    if yield_for_recommended is not None:
        diff_pct = (yield_for_recommended - predicted_yield) / max(predicted_yield, 1e-6) * 100.0
        if diff_pct > 8:
            tips.append(
                f"Switching to the recommended crop could lift estimated yield by about {diff_pct:.0f}% "
                "under similar farm conditions (model estimate)."
            )
        elif diff_pct < -8:
            tips.append(
                "Your current crop shows a higher modeled yield than the recommended one for these conditions; "
                "validate with local agronomy advice."
            )

    tips.append(
        "Improvement levers: optimize planting dates, soil moisture, balanced NPK based on tests, "
        "and pest or disease scouting."
    )
    return tips
