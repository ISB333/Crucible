"""Artifact model: a workspace of files with delimited editable regions (PRD §4)."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

START_RE = re.compile(r"crucible:region start name=([A-Za-z0-9_-]+)")
END_RE = re.compile(r"crucible:region end")


class RegionError(ValueError):
    """Malformed region markers (unclosed, nested, end-without-start)."""


@dataclass(frozen=True)
class Region:
    file: str
    name: str
    start: int  # index of first editable line (the line after the start marker)
    end: int  # index of the end-marker line (exclusive bound of editable lines)


@dataclass(frozen=True)
class Hole:
    file: str
    line: int  # 0-based
    kind: str  # "sentinel" | "not_implemented" | verifier-specific (e.g. "sorry")
    text: str


def parse_regions(file: str, text: str) -> tuple[Region, ...]:
    regions: list[Region] = []
    open_at: tuple[str, int] | None = None  # (name, first editable line)
    for i, line in enumerate(text.splitlines()):
        if m := START_RE.search(line):
            if open_at is not None:
                raise RegionError(f"{file}:{i + 1}: nested region start")
            open_at = (m.group(1), i + 1)
        elif END_RE.search(line):
            if open_at is None:
                raise RegionError(f"{file}:{i + 1}: region end without start")
            regions.append(Region(file=file, name=open_at[0], start=open_at[1], end=i))
            open_at = None
    if open_at is not None:
        raise RegionError(f"{file}: unclosed region {open_at[0]!r}")
    return tuple(regions)


@dataclass(frozen=True)
class Artifact:
    files: Mapping[str, str]  # relative path -> text, normalized to trailing newline
    regions: tuple[Region, ...]
    parent_hash: str | None = None

    @classmethod
    def from_files(cls, files: Mapping[str, str], parent_hash: str | None = None) -> "Artifact":
        def _normalize(text: str) -> str:
            lf = text.replace("\r\n", "\n").replace("\r", "\n")
            return lf if lf.endswith("\n") or not lf else lf + "\n"

        normalized = {path: _normalize(text) for path, text in files.items()}
        regions = tuple(
            r for path in sorted(normalized) for r in parse_regions(path, normalized[path])
        )
        return cls(MappingProxyType(normalized), regions, parent_hash)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for path in sorted(self.files):
            h.update(path.encode())
            h.update(b"\0")
            h.update(self.files[path].encode())
            h.update(b"\0")
        return h.hexdigest()

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"no editable region named {name!r}")

    def region_text(self, r: Region) -> str:
        return "\n".join(self.files[r.file].splitlines()[r.start : r.end])

    def replace_region(self, name: str, new_text: str) -> "Artifact":
        r = self.region(name)
        lines = self.files[r.file].splitlines()
        spliced = lines[: r.start] + new_text.splitlines() + lines[r.end :]
        files = dict(self.files) | {r.file: "\n".join(spliced) + "\n"}
        return Artifact.from_files(files, parent_hash=self.content_hash)


HOLE_SENTINEL = "crucible:hole"
NOT_IMPLEMENTED_RE = re.compile(r"raise\s+NotImplementedError")


def masked_view(a: Artifact) -> dict[str, str]:
    """Files with each editable region's body replaced by a placeholder.

    Two artifacts are identical outside their editable regions iff their
    masked views are equal (integrity check 1, PRD §7).
    """
    out: dict[str, str] = {}
    for path, text in a.files.items():
        lines = text.splitlines()
        per_file = sorted(
            (r for r in a.regions if r.file == path), key=lambda r: r.start, reverse=True
        )
        for r in per_file:
            lines[r.start : r.end] = [f"\x00region:{r.name}\x00"]
        out[path] = "\n".join(lines)
    return out


def scan_holes(a: Artifact) -> tuple[Hole, ...]:
    """Generic holes: the crucible:hole sentinel and raise NotImplementedError."""
    holes: list[Hole] = []
    for path in sorted(a.files):
        for i, line in enumerate(a.files[path].splitlines()):
            if HOLE_SENTINEL in line:
                holes.append(Hole(file=path, line=i, kind="sentinel", text=line.strip()))
            elif NOT_IMPLEMENTED_RE.search(line):
                holes.append(Hole(file=path, line=i, kind="not_implemented", text=line.strip()))
    return tuple(holes)
