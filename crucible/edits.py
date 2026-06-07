"""Region-enforced edit application (PRD §5): the tool enforces
the boundary, not trust.
"""

from dataclasses import dataclass

from crucible.artifact import Artifact

MARKER_TOKEN = "crucible:region"


@dataclass(frozen=True)
class EditResult:
    artifact: Artifact  # the new artifact if applied, the input artifact if not
    applied: bool
    error: str | None = None


def search_replace(a: Artifact, file: str, old: str, new: str) -> EditResult:
    text = a.files.get(file)
    if text is None:
        return EditResult(a, False, f"unknown file {file!r}")
    if MARKER_TOKEN in old or MARKER_TOKEN in new:
        return EditResult(a, False, "edits may not touch region markers — rejected")
    count = text.count(old)
    if count == 0:
        return EditResult(a, False, "old text not found in file")
    if count > 1:
        return EditResult(
            a,
            False,
            f"old text matches {count} times; provide a unique snippet",
        )
    start = text.index(old)
    line_lo = text.count("\n", 0, start)
    line_hi = line_lo + old.count("\n")
    if old.endswith("\n"):  # trailing newline terminates the last matched line
        line_hi -= 1
    inside = any(r.file == file and r.start <= line_lo and line_hi < r.end for r in a.regions)
    if not inside:
        return EditResult(a, False, "edit outside editable regions — rejected")
    files = dict(a.files) | {file: text.replace(old, new, 1)}
    return EditResult(Artifact.from_files(files, parent_hash=a.content_hash), True)


def write_region(a: Artifact, name: str, content: str) -> EditResult:
    if MARKER_TOKEN in content:
        return EditResult(a, False, "edits may not touch region markers — rejected")
    try:
        return EditResult(a.replace_region(name, content), True)
    except KeyError as e:
        return EditResult(a, False, str(e))
