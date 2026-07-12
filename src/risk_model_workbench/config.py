"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_MERGE_KEY_SENTINEL = object()


class ConfigError(ValueError):
    """Base class for user-correctable configuration errors."""


class ConfigFormatError(ConfigError):
    """Raised when a YAML config does not satisfy the mapping contract."""

    def __init__(self, path: str | Path, detail: str):
        self.path = Path(path)
        self.detail = detail
        super().__init__(f"invalid YAML config {self.path}: {detail}")


class ConfigConflictError(ConfigError):
    """Raised when canonical and legacy config variants have diverged."""

    def __init__(self, canonical: str | Path, legacy: str | Path):
        self.canonical = Path(canonical)
        self.legacy = Path(legacy)
        super().__init__(
            "configuration variants diverge: "
            f"canonical={self.canonical}; legacy={self.legacy}"
        )


def _strict_loader(yaml_module):
    class StrictMappingLoader(yaml_module.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            is_merge_key = key_node.tag == "tag:yaml.org,2002:merge"
            key = (
                _MERGE_KEY_SENTINEL
                if is_merge_key
                else loader.construct_object(key_node, deep=False)
            )
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise yaml_module.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                label = "merge '<<'" if is_merge_key else repr(key)
                raise yaml_module.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {label}",
                    key_node.start_mark,
                )
            seen.add(key)
        return yaml_module.SafeLoader.construct_mapping(loader, node, deep=deep)

    StrictMappingLoader.add_constructor(
        yaml_module.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return StrictMappingLoader


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load exactly one YAML mapping without interpolating string values."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyYAML is required to read project configs. Install with: pip install pyyaml"
        ) from exc

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            documents = list(yaml.load_all(handle, Loader=_strict_loader(yaml)))
    except yaml.YAMLError as exc:
        raise ConfigFormatError(config_path, str(exc)) from exc
    if len(documents) > 1:
        raise ConfigFormatError(config_path, "expected exactly one YAML document")
    data = documents[0] if documents else None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigFormatError(config_path, "expected a YAML mapping")
    return data


def resolve_yaml_variant(canonical: str | Path, legacy: str | Path) -> Path:
    """Resolve a canonical/mirror pair and fail closed when both diverge."""
    canonical_path = Path(canonical)
    legacy_path = Path(legacy)
    canonical_exists = canonical_path.exists()
    legacy_exists = legacy_path.exists()
    if canonical_exists and legacy_exists:
        try:
            canonical_data = load_yaml(canonical_path)
            legacy_data = load_yaml(legacy_path)
        except ConfigFormatError as exc:
            raise ConfigFormatError(
                exc.path,
                f"{exc.detail}; configuration pair: canonical={canonical_path}; legacy={legacy_path}",
            ) from exc
        if canonical_data != legacy_data:
            raise ConfigConflictError(canonical_path, legacy_path)
        return canonical_path
    if canonical_exists:
        load_yaml(canonical_path)
        return canonical_path
    if legacy_exists:
        load_yaml(legacy_path)
        return legacy_path
    return canonical_path


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a YAML mapping."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyYAML is required to write project configs. Install with: pip install pyyaml"
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
