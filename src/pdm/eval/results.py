"""Run records.

The project rule is that no figure or table may come from an unrecorded run.
That only holds if a record captures enough to reproduce the number: the full
configuration, the seeds, the exact code, and the library versions.

The code identity is the part most often got wrong. A commit SHA identifies the
code only if the working tree was clean when the run happened; otherwise it names
a commit that does not contain what actually ran. `GitState` therefore records
`dirty` alongside the SHA, and `ResultsWriter` warns when a run is recorded from
a dirty tree.
"""

from __future__ import annotations

import json
import platform
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitState:
    """Which code produced a result."""

    sha: str | None
    branch: str | None
    dirty: bool

    @property
    def identifies_the_code(self) -> bool:
        """True only when the SHA fully determines what ran."""
        return self.sha is not None and not self.dirty

    @classmethod
    def capture(cls, repo_root: Path | str | None = None) -> GitState:
        cwd = str(repo_root) if repo_root else None

        def git(*args: str) -> str | None:
            try:
                out = subprocess.run(
                    ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
                )
            except (subprocess.SubprocessError, OSError):
                return None
            return out.stdout.strip()

        sha = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        status = git("status", "--porcelain")
        return cls(sha=sha, branch=branch, dirty=bool(status))


def environment() -> dict[str, str]:
    """Library versions that can move a numeric result between machines."""
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            versions[module] = __import__(module).__version__
        except (ImportError, AttributeError):  # pragma: no cover - environment dependent
            versions[module] = "not installed"
    return versions


@dataclass
class RunRecord:
    """One run, fully described.

    `metrics` holds whatever the run produced -- typically `CVResult.to_dict()`.
    Everything else exists so the metrics can be trusted and reproduced.
    """

    name: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    seeds: list[int]
    git: GitState
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    environment: dict[str, str] = field(default_factory=environment)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_utc": self.created_utc,
            "git": asdict(self.git),
            "reproducible": self.git.identifies_the_code,
            "seeds": list(self.seeds),
            "environment": self.environment,
            "config": self.config,
            "metrics": self.metrics,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)


class ResultsWriter:
    """Writes run records to `results/`, one JSON per run.

    Filenames carry the run name and a UTC timestamp, so repeated runs of the
    same config accumulate rather than overwrite. Results are committed: they
    are the audit trail, not regenerable output.
    """

    def __init__(self, results_dir: Path | str, warn_when_dirty: bool = True) -> None:
        self.results_dir = Path(results_dir)
        self.warn_when_dirty = warn_when_dirty

    def write(self, record: RunRecord) -> Path:
        if self.warn_when_dirty and not record.git.identifies_the_code:
            warnings.warn(
                f"run '{record.name}' recorded from a dirty or non-git tree: its SHA does not "
                "identify the code that produced these numbers. Commit before any run whose "
                "output reaches the report.",
                stacklevel=2,
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = record.created_utc.replace(":", "").replace("-", "")
        path = self.results_dir / f"{record.name}__{stamp}.json"
        path.write_text(record.to_json(), encoding="utf-8")
        return path
