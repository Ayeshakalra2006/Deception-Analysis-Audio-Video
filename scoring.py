
from interview_config import SCORING_WEIGHTS


def calculate_session_score(
    contradictions,
    voice_deviation=0.0,
    model_score=0.50
):
    """
    Combines experimental indicators into a session score.

    All scores are expected between 0 and 1.
    """

    contradiction_score = min(len(contradictions) / 3, 1.0)

    voice_deviation = max(0.0, min(voice_deviation, 1.0))
    model_score = max(0.0, min(model_score, 1.0))

    weights = SCORING_WEIGHTS

    final_score = (
        contradiction_score * weights["contradictions"]
        + voice_deviation * weights["voice_deviation"]
        + model_score * weights["model_score"]
    )

    final_score = round(final_score * 100, 2)

    if final_score < 35:
        label = "Likely consistent"
    elif final_score < 65:
        label = "Some inconsistencies"
    else:
        label = "High risk indicators"

    return {
        "score": final_score,
        "label": label,
        "contradiction_score": round(contradiction_score * 100, 2),
        "voice_deviation": round(voice_deviation * 100, 2),
        "model_score": round(model_score * 100, 2)
    }
