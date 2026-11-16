
import re


def extract_time(text):
    """
    Extract time in minutes from answers:
    8 AM, 8:30 am, 19:00, 6 in the morning.
    """

    if not text:
        return None

    text = str(text).lower().strip()

    patterns = [
        r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
        r'\b(\d{1,2})\s*(am|pm)\b',
        r'\b(\d{1,2})\s*(?:in the morning|in morning)\b',
        r'\b(\d{1,2})\s*(?:in the evening|in evening|at night)\b'
    ]

    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text)

        if not match:
            continue

        hour = int(match.group(1))

        if index == 0:
            minute = int(match.group(2))
            period = match.group(3)
        else:
            minute = 0
            period = match.group(2) if index == 1 else None

            if index == 2:
                period = "am"
            elif index == 3:
                period = "pm"

        if minute > 59:
            return None

        if period:
            if hour < 1 or hour > 12:
                return None

            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
        elif hour > 23:
            return None

        return hour * 60 + minute

    return None


def format_time(minutes):
    """Convert minutes into readable 12-hour time."""

    hour = (minutes // 60) % 24
    minute = minutes % 60

    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12

    if display_hour == 0:
        display_hour = 12

    return f"{display_hour}:{minute:02d} {period}"


def extract_facts(answers):
    """
    Extract facts from stored interview answers.

    Expected format:
    {
        "wake_time": "I woke up at 8 AM",
        "breakfast": "I had breakfast at 6 AM"
    }
    """

    facts = {
        "wake_time": [],
        "breakfast_time": []
    }

    for question_id, answer in answers.items():

        if not answer:
            continue

        time_value = extract_time(answer)

        if time_value is None:
            continue

        question_id = str(question_id).lower()

        if "wake" in question_id:
            facts["wake_time"].append({
                "time": time_value,
                "answer": str(answer),
                "question_id": question_id
            })

        elif "breakfast" in question_id:
            facts["breakfast_time"].append({
                "time": time_value,
                "answer": str(answer),
                "question_id": question_id
            })

    return facts


def detect_contradictions(facts):
    """
    Detect contradictions between repeated answers
    and logically conflicting morning timelines.
    """

    contradictions = []

    wake_times = facts.get("wake_time", [])
    breakfast_times = facts.get("breakfast_time", [])

    # Compare ALL repeated wake-up answers.
    for i in range(len(wake_times)):
        for j in range(i + 1, len(wake_times)):

            first = wake_times[i]["time"]
            second = wake_times[j]["time"]

            if abs(first - second) > 60:
                contradictions.append(
                    f"Wake-up time inconsistency: "
                    f"You first said {format_time(first)}, "
                    f"but later said {format_time(second)}."
                )

    # Compare ALL repeated breakfast answers.
    for i in range(len(breakfast_times)):
        for j in range(i + 1, len(breakfast_times)):

            first = breakfast_times[i]["time"]
            second = breakfast_times[j]["time"]

            if abs(first - second) > 60:
                contradictions.append(
                    f"Breakfast time inconsistency: "
                    f"You first said {format_time(first)}, "
                    f"but later said {format_time(second)}."
                )

    # Check breakfast against wake-up time.
    for wake in wake_times:
        for breakfast in breakfast_times:

            if breakfast["time"] < wake["time"]:

                contradictions.append(
                    f"Timeline contradiction: "
                    f"You said you woke up at "
                    f"{format_time(wake['time'])}, "
                    f"but reported having breakfast at "
                    f"{format_time(breakfast['time'])}."
                )

    return contradictions
