"""Stateless argument and deadline checks for Cron-bound camera commands."""

import re


def semantic_arguments_allowed(arguments: list[str]) -> bool:
    if not arguments or len(arguments) % 2 != 1:
        return False
    fields = dict(zip(arguments[1::2], arguments[2::2]))
    if len(fields) != (len(arguments) - 1) // 2:
        return False
    if arguments[0] == "observe":
        return (
            set(fields) == {"--policy-id", "--timeout-seconds"}
            and re.fullmatch(r"[1-9][0-9]{0,9}", fields["--timeout-seconds"]) is not None
            and int(fields["--timeout-seconds"]) <= 2**31 - 1
            and semantic_arguments_allowed(["candidate", "--policy-id", fields["--policy-id"]])
        )
    if arguments[0] == "candidate":
        return set(fields) == {"--policy-id"} and re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            fields["--policy-id"],
        ) is not None
    if arguments[0] != "commit" or not re.fullmatch(
        r"[A-Za-z0-9_-]{32,128}", fields.get("--capability", "")
    ):
        return False
    required = {"--capability", "--matched"}
    if fields.get("--matched") == "false":
        return set(fields) <= required | {"--unknown"} and fields.get("--unknown", "false") in {"true", "false"}
    required |= {"--subject-kind", "--predicate", "--duration-seconds", "--evidence-ref"}
    if (
        fields.get("--matched") != "true"
        or not required <= fields.keys()
        or fields.keys() - required - {"--subject-ref", "--zone-id", "--frame-states", "--track-ids", "--frame-positions", "--view-aligned"}
        or fields["--subject-kind"] not in {"person", "object"}
        or fields["--predicate"] not in {"appears", "disappears", "enters_zone", "leaves_zone", "lingers"}
        or re.fullmatch(r"[0-9]{1,4}", fields["--duration-seconds"]) is None
        or int(fields["--duration-seconds"]) > 3600
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", fields["--evidence-ref"]) is None
    ):
        return False
    geometry = "--frame-positions" in fields or "--view-aligned" in fields
    if geometry and (not fields.get("--zone-id") or "--frame-states" not in fields or "--track-ids" not in fields):
        return False
    if "--frame-states" in fields or "--track-ids" in fields:
        states = fields.get("--frame-states", "").split(",")
        tracks = fields.get("--track-ids", "").split(",")
        if (
            not 3 <= len(states) <= 8 or len(states) != len(tracks)
            or any(state not in {"present", "absent", "inside", "outside", "unknown"} for state in states)
            or any(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", track) is None for track in tracks)
        ):
            return False
        if geometry:
            positions = fields.get("--frame-positions", "")
            aligned = fields.get("--view-aligned", "")
            if len(positions) > 200 or len(aligned) > 48:
                return False
            positions, aligned = positions.split(","), aligned.split(",")
            coordinate = r"(?:0(?:\.[0-9]{1,9})?|1(?:\.0{1,9})?)"
            if (len(positions) != len(states) or len(aligned) != len(states)
                    or any(value not in {"true", "false"} for value in aligned)
                    or any(point != "null" and re.fullmatch(coordinate + ":" + coordinate, point) is None for point in positions)):
                return False
    return all(
        len(fields.get(key, "").encode("utf-8")) <= limit
        and not any(ord(c) < 0x20 for c in fields.get(key, ""))
        for key, limit in (("--subject-ref", 120), ("--zone-id", 64))
    )


def semantic_execution_timeout(arguments: list[str], requested: int, foreground_limit: int, ordinary_limit: int) -> int:
    if not arguments or arguments[0] != "observe":
        return max(1, min(requested, ordinary_limit))
    if not semantic_arguments_allowed(arguments):
        raise ValueError("Invalid finite observation command")
    fields = dict(zip(arguments[1::2], arguments[2::2]))
    budget = int(fields["--timeout-seconds"])
    # Server closes at budget; HTTP has 2s and process has 5s to return a result.
    # Never silently clamp an observation into a shorter capture interval.
    if budget + 5 > min(requested, foreground_limit):
        raise ValueError("Observation execution budget exceeds the foreground timeout")
    return budget + 5
