"""Sandbox protocol (PRD §3: sandboxed, bounded). Implementations land in Plan 2."""

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class Sandbox(Protocol):
    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_s: float, network: bool = False
    ) -> SandboxResult: ...


def _text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


@dataclass(frozen=True)
class SubprocessSandbox:
    """Timeout + minimal env only — NO isolation. Dev/tests; real runs use DockerSandbox."""

    env_passthrough: tuple[str, ...] = ("PATH", "HOME", "LANG")

    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_s: float, network: bool = False
    ) -> SandboxResult:
        env = {k: os.environ[k] for k in self.env_passthrough if k in os.environ}
        try:
            p = subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            return SandboxResult(124, _text(e.stdout), _text(e.stderr), timed_out=True)
        except FileNotFoundError as e:
            return SandboxResult(127, "", str(e))
        except PermissionError as e:
            return SandboxResult(126, "", str(e))  # found but not executable (shell convention)
        except OSError as e:
            return SandboxResult(125, "", str(e))  # other spawn failure (bad cwd, …)
        return SandboxResult(p.returncode, _text(p.stdout), _text(p.stderr))


@dataclass(frozen=True)
class DockerSandbox:
    """One `docker run` per invocation: no host FS beyond the workspace, no network
    unless the task declares it, memory/cpu caps (PRD §3 required properties).
    Cold image pulls count toward timeout_s — pre-pull images for tight budgets."""

    image: str = "python:3.12-slim"
    memory: str = "1g"
    cpus: str = "1"
    user: str | None = None

    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_s: float, network: bool = False
    ) -> SandboxResult:
        name = f"crucible-{uuid4().hex[:12]}"
        user_spec = self.user or f"{os.getuid()}:{os.getgid()}"
        # First, ensure the /work directory has the correct permissions
        chown_argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"{name}-chown",
            "--network",
            "bridge" if network else "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "-v",
            f"{cwd.resolve()}:/work",
            "-w",
            "/work",
            self.image,
            "chown",
            "-R",
            user_spec,
            "/work",
        ]
        try:
            subprocess.run(chown_argv, capture_output=True, timeout=timeout_s)
        except Exception:
            pass
        # Now run the main command with the correct user
        docker_argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "bridge" if network else "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "-v",
            f"{cwd.resolve()}:/work",
            "-w",
            "/work",
            "--user",
            user_spec,
            self.image,
            *argv,
        ]
        try:
            p = subprocess.run(docker_argv, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            # kill may race a not-yet-started container; --rm is the cleanup backstop
            return SandboxResult(124, _text(e.stdout), _text(e.stderr), timed_out=True)
        except FileNotFoundError as e:  # docker not installed — degrade, don't crash
            return SandboxResult(127, "", str(e))
        except PermissionError as e:
            return SandboxResult(126, "", str(e))  # found but not executable (shell convention)
        except OSError as e:
            return SandboxResult(125, "", str(e))  # other spawn failure (bad cwd, …)
        # If we ran as root, change ownership of all files in /work back to the host user's UID:GID
        if self.user == "root" or user_spec == "root:root":
            original_user_spec = f"{os.getuid()}:{os.getgid()}"
            chown_back_argv = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"{name}-chown-back",
                "--network",
                "bridge" if network else "none",
                "--memory",
                self.memory,
                "--cpus",
                self.cpus,
                "-v",
                f"{cwd.resolve()}:/work",
                "-w",
                "/work",
                self.image,
                "chown",
                "-R",
                original_user_spec,
                "/work",
            ]
            try:
                subprocess.run(chown_back_argv, capture_output=True, timeout=timeout_s)
            except Exception:
                pass
        return SandboxResult(p.returncode, _text(p.stdout), _text(p.stderr))
