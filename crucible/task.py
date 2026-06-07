"""Task spec: which files form the workspace and which regions are editable."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    root: Path
    files: tuple[str, ...]  # relative paths, sorted
    editable: tuple[str, ...]  # region names the agent may edit
    network: bool = False  # verifier may use the network iff True (PRD §3)
    deny_tokens: tuple[str, ...] = ()  # extra escape tokens beyond the defaults (PRD §7)

    @classmethod
    def from_path(
        cls, path: str | Path, editable: list[str] | tuple[str, ...], network: bool = False
    ) -> "Task":
        p = Path(path).resolve()
        if p.is_file():
            return cls(root=p.parent, files=(p.name,), editable=tuple(editable), network=network)
        files = tuple(
            sorted(
                str(f.relative_to(p))
                for f in p.rglob("*")
                if f.is_file() and not any(part.startswith(".") for part in f.relative_to(p).parts)
            )
        )
        return cls(root=p, files=files, editable=tuple(editable), network=network)

    def load_files(self) -> dict[str, str]:
        return {rel: (self.root / rel).read_text() for rel in self.files}
