"""Reference verifiers (PRD §3). Import from submodules; re-exported here as they land."""

from crucible.verifiers.chem_verifier import Chem
from crucible.verifiers.command import Command
from crucible.verifiers.forge_verifier import Forge
from crucible.verifiers.lean import Lean
from crucible.verifiers.pyright_verifier import Pyright
from crucible.verifiers.pytest_verifier import Pytest
from crucible.verifiers.rubric import Rubric

__all__ = ["Chem", "Command", "Forge", "Lean", "Pyright", "Pytest", "Rubric"]
