"""Canonical, immutable Agent action invocations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
CanonicalValue: TypeAlias = JsonScalar | tuple["CanonicalValue", ...] | Mapping[str, "CanonicalValue"]


@dataclass(frozen=True)
class ActionInvocation:
    tool_name: str
    params: Mapping[str, CanonicalValue]
    project: str
    version_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("project must be a non-empty string")
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ValueError("version_id must be a non-empty string")
        frozen = _freeze(self.params, path="params")
        if not isinstance(frozen, Mapping):
            raise TypeError("params must be a mapping")
        object.__setattr__(self, "params", frozen)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionInvocation":
        raw_params = payload.get("params")
        if not isinstance(raw_params, Mapping):
            raise TypeError("params must be a mapping")
        return cls(
            tool_name=str(payload.get("tool_name") or ""),
            params=raw_params,
            project=str(payload.get("project") or ""),
            version_id=str(payload.get("version_id") or ""),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "params": _thaw(self.params),
            "project": self.project,
            "version_id": self.version_id,
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical_payload()).encode("utf-8")).hexdigest()


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _freeze(value: object, *, path: str) -> CanonicalValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        items: dict[str, CanonicalValue] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            items[key] = _freeze(value[key], path=f"{path}.{key}")
        return MappingProxyType(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, path=f"{path}[]") for item in value)
    raise TypeError(f"{path} contains unsupported value: {type(value).__name__}")


def _thaw(value: CanonicalValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
