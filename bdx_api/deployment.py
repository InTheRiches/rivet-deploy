"""Shared, editable hardware mapping for Rivet deployment and commissioning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOYMENT_CONFIG = PROJECT_ROOT / "config" / "rivet_deployment.yaml"


def resolve_deployment_config(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv("RIVET_DEPLOYMENT_CONFIG")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DEPLOYMENT_CONFIG


def load_deployment_config(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_deployment_config(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Deployment config must contain a YAML mapping: {config_path}")

    channels = data.get("can_channels")
    joints = data.get("joints")
    if not isinstance(channels, list) or not channels or len(set(channels)) != len(channels):
        raise ValueError("can_channels must be a non-empty list of unique CAN interface names")
    if not isinstance(joints, dict) or not joints:
        raise ValueError("joints must map joint names to motor wiring")

    used_ids: dict[int, str] = {}
    for name, entry in joints.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Wiring for {name} must be a YAML mapping")
        channel = entry.get("can_interface")
        motor_id = entry.get("motor_id")
        motor_type = entry.get("motor_type")
        direction = entry.get("direction", 1)
        if channel not in channels:
            raise ValueError(f"{name} uses unknown CAN interface {channel!r}")
        if not isinstance(motor_id, int) or not 1 <= motor_id <= 127:
            raise ValueError(f"{name} motor_id must be between 1 and 127")
        if motor_id in used_ids:
            raise ValueError(
                f"Duplicate motor ID {motor_id} assigned to {used_ids[motor_id]} and {name}; "
                "motor IDs must be unique across this robot's CAN buses"
            )
        used_ids[motor_id] = name
        if motor_type not in {"O2", "O3", "O5", "O6"}:
            raise ValueError(f"{name} has unsupported motor_type {motor_type!r}")
        if direction not in (-1, 1):
            raise ValueError(f"{name} direction must be 1 or -1")

    return config_path, data


def save_deployment_config(path: str | Path, data: dict[str, Any]) -> None:
    """Write config atomically so a partial process exit cannot truncate it."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(config_path)
