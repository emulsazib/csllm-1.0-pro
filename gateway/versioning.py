"""Versioned model configurations.

``POST /configure_model`` writes a new config version rather than mutating one,
so a checkpoint can always be traced back to the exact hyperparameters that
produced it.

Validation is delegated to the **C++** ``ModelConfig.validate()``. That is the
single source of truth for invariants like "head_dim must be even for RoPE"
(``core/src/model.cpp``); duplicating them in Python would let the two drift and
allow a config the engine then rejects at build time.

Version ids are ``v{N}-{hash}`` where the hash is over the canonical config, so
identical hyperparameters resolve to the same id and re-submitting is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csllm import _csllm_core as core
from csllm.config import config_from_dict, config_to_dict

__all__ = ["ConfigVersion", "ConfigStore"]

VERSIONS_DIR = Path("configs/versions")


@dataclass(frozen=True)
class ConfigVersion:
    version_id: str
    index: int
    config: dict[str, Any]
    num_params: int
    activation_bytes: int
    created_at: float
    path: Path
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "index": self.index,
            "config": self.config,
            "num_params": self.num_params,
            "activation_bytes": self.activation_bytes,
            "created_at": self.created_at,
            "path": str(self.path),
            "note": self.note,
        }


def _fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


class ConfigStore:
    def __init__(self, directory: str | Path = VERSIONS_DIR) -> None:
        self.directory = Path(directory)

    # ── read ─────────────────────────────────────────────────────────────────

    def list(self) -> list[ConfigVersion]:
        if not self.directory.is_dir():
            return []
        versions = []
        for path in sorted(self.directory.glob("v*.json")):
            try:
                versions.append(self._load_file(path))
            except (json.JSONDecodeError, KeyError):
                continue  # a hand-edited file should not break the listing
        return sorted(versions, key=lambda v: v.index)

    def get(self, version_id: str) -> ConfigVersion:
        path = self.directory / f"{version_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no such config version: {version_id}")
        return self._load_file(path)

    @staticmethod
    def _load_file(path: Path) -> ConfigVersion:
        payload = json.loads(path.read_text())
        return ConfigVersion(
            version_id=payload["version_id"],
            index=payload["index"],
            config=payload["config"],
            num_params=payload["num_params"],
            activation_bytes=payload["activation_bytes"],
            created_at=payload["created_at"],
            path=path,
            note=payload.get("note", ""),
        )

    def _next_index(self) -> int:
        existing = self.list()
        return (max((v.index for v in existing), default=0)) + 1

    # ── write ────────────────────────────────────────────────────────────────

    def create(
        self,
        config: dict[str, Any],
        note: str = "",
        batch_size: int = 8,
    ) -> tuple[ConfigVersion, bool]:
        """Validate and persist a config. Returns (version, created_now).

        Re-submitting identical hyperparameters returns the existing version
        instead of piling up duplicates.
        """
        # C++ owns the invariants; this raises RuntimeError on a bad config.
        cfg = config_from_dict(config)
        canonical = config_to_dict(cfg)

        fingerprint = _fingerprint(canonical)
        for existing in self.list():
            if existing.version_id.endswith(f"-{fingerprint}"):
                return existing, False

        index = self._next_index()
        version_id = f"v{index}-{fingerprint}"
        version = ConfigVersion(
            version_id=version_id,
            index=index,
            config=canonical,
            num_params=cfg.num_params(),
            activation_bytes=core.estimate_activation_bytes(cfg, batch_size, cfg.block_size),
            created_at=time.time(),
            path=self.directory / f"{version_id}.json",
            note=note,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        version.path.write_text(json.dumps(version.to_dict(), indent=1))
        return version, True

    def build_model(self, version: ConfigVersion, out_path: str | Path, seed: int = 1337) -> Path:
        """Initialize a fresh model at this config and write a checkpoint."""
        cfg = config_from_dict(version.config)
        model = core.Model(cfg, seed)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        return out
