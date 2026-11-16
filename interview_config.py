
# Interview questions and scoring configuration

QUESTIONS = [
    {
        "id": "intro",
        "question": "Please introduce yourself.",
        "fact": None,
        "type": "baseline"
    },
    {
        "id": "wake_time",
        "question": "What time did you wake up today?",
        "fact": "wake_time",
        "type": "fact"
    },
    {
        "id": "breakfast",
        "question": "What did you have for breakfast, and at what time?",
        "fact": "breakfast",
        "type": "fact"
    },
    {
        "id": "daily_routine",
        "question": "Can you describe what you did after waking up?",
        "fact": "morning_routine",
        "type": "fact"
    },
    {
        "id": "wake_time_repeat",
        "question": "Just to confirm, approximately what time did you get out of bed today?",
        "fact": "wake_time",
        "type": "repeat"
    },
    {
        "id": "breakfast_repeat",
        "question": "Before or after waking up, when did you have your breakfast?",
        "fact": "breakfast",
        "type": "repeat"
    }
]

# All scoring weights are configurable here.
SCORING_WEIGHTS = {
    "contradictions": 0.50,
    "voice_deviation": 0.20,
    "model_score": 0.30
}

# Used for Stage 1 testing until real model integration.
DUMMY_MODEL_SCORE = 0.50
