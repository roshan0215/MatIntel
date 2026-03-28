from __future__ import annotations

import os


def generate_material_summary(row: dict, top_application: str) -> str:
    """Generate a short explanation, using Anthropic if configured or local fallback otherwise."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_summary(row, top_application)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are a materials scientist. Explain this material to a product engineer.\n\n"
            f"Material: {row.get('Reduced Formula', 'unknown')}\n"
            f"Formation energy: {row.get('Formation Energy Per Atom', 'unknown')} eV/atom\n"
            f"Band gap: {row.get('Bandgap', 'unknown')} eV\n"
            f"Top application: {top_application}\n"
            f"Application score: {row.get('selected_score', 'unknown')}\n"
            f"Viability score: {row.get('viability', 'unknown')}\n\n"
            "Write 3 sentences:\n"
            "1) Why this material is promising for the selected application.\n"
            "2) Main limitation or trade-off.\n"
            "3) What to verify next in the lab.\n"
            "Use plain English and include specific numbers when available."
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=240,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        return _fallback_summary(row, top_application)


def _fallback_summary(row: dict, top_application: str) -> str:
    formula = row.get("Reduced Formula", "unknown")
    band_gap = row.get("Bandgap", "unknown")
    viability = row.get("viability", "unknown")
    score = row.get("selected_score", "unknown")
    return (
        f"{formula} scores {score} for {top_application} with viability {viability}. "
        f"Its band gap is {band_gap} eV, which is a key screening metric for this use case. "
        "Next, validate synthesis feasibility and test experimental performance under operating conditions."
    )
