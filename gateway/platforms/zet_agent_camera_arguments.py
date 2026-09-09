"""Stateless camera history argument validation for the trusted script runner."""

from datetime import datetime, timedelta, timezone
import re


def camera_arguments_allowed(arguments: list[str]) -> bool:
    if arguments == ["list"]:
        return True
    if arguments and arguments[0] == "history":
        return history_arguments_allowed(arguments)
    if len(arguments) not in (3, 5) or arguments[1] != "--camera-id":
        return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", arguments[2]) is None:
        return False
    if len(arguments) == 3:
        return arguments[0] in {"snap", "doctor", "clip"}
    return arguments[0] == "clip" and arguments[3] == "--duration" and arguments[4].isdigit() and 1 <= int(arguments[4]) <= 60


def history_arguments_allowed(arguments: list[str]) -> bool:
    if len(arguments) != 7 or arguments[0] != "history":
        return False
    if arguments[1::2] != ["--camera-id", "--start", "--end"]:
        return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", arguments[2]) is None:
        return False
    times = []
    for value in (arguments[4], arguments[6]):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value) is None:
            return False
        try:
            times.append(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc))
        except (ValueError, OverflowError):
            return False
    return timedelta(milliseconds=1) <= times[1] - times[0] <= timedelta(seconds=30) and times[1] <= datetime.now(timezone.utc)
