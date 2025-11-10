from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from jinja2 import Environment, StrictUndefined, Template

from config import PromptTemplateConfig
from ..utils import clean_text

logger = logging.getLogger(__name__)

def _create_environment() -> Environment:
    env = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    env.filters["clean"] = clean_text
    env.globals["clean_text"] = clean_text
    return env


def _normalise_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
    for key, value in mapping.items():
        try:
            str_key = str(key)
        except Exception:
            str_key = repr(key)
        normalised[str_key] = value
    return normalised


def _clean_mapping(mapping: Mapping[str, Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for key, value in mapping.items():
        cleaned[key] = clean_text(value)
    return cleaned


def _item_fields(columns: Mapping[str, Any]) -> Dict[str, str]:
    cleaned = _clean_mapping(columns)
    return {
        key: value
        for key, value in cleaned.items()
        if not key.startswith("_")
    }


def _load_template_text(
    *,
    cfg: PromptTemplateConfig,
    template_value: Optional[str],
    template_path: Optional[str],
    base_dir: Optional[Path],
    encoding: str,
    label: str,
) -> str:
    if template_value:
        return template_value

    if template_path:
        try:
            resolved = cfg.resolve_path(template_path, base_dir=base_dir)
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to resolve prompt template path: {template_path}"
            ) from exc

        if not resolved.exists():
            raise FileNotFoundError(
                f"Prompt template file not found: {resolved}"
            )

        logger.debug("Loading %s prompt template from %s", label, resolved)
        return resolved.read_text(encoding=encoding)

    raise ValueError(
        f"Prompt configuration is missing a {label} template. "
        "Provide either an inline template or a template path."
    )


@dataclass
class PromptRenderer:
    """Render chat completion messages from prompt templates."""

    system_template: Template
    user_template: Template

    @classmethod
    def from_strings(cls, system_template: str, user_template: str) -> "PromptRenderer":
        env = _create_environment()
        system = env.from_string(system_template.strip())
        user = env.from_string(user_template.strip())
        return cls(system_template=system, user_template=user)

    def render_messages(
        self,
        tree_markdown: str,
        item: Mapping[str, Any],
        *,
        item_columns: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, str]]:
        item_data = _normalise_mapping(item)
        item_clean = _clean_mapping(item_data)

        columns = _normalise_mapping(item_columns or {})
        columns_clean = _clean_mapping(columns) if columns else {}
        item_fields = _item_fields(columns) if columns else {}
        context = {
            "tree": (tree_markdown or "").strip(),
            "item": item_data,
            "item_clean": item_clean,
            "item_columns": columns,
            "item_columns_clean": columns_clean,
            "columns": columns_clean,
            "item_fields": item_fields,
        }

        system_content = self.system_template.render(context).strip()
        user_content = self.user_template.render(context).strip()
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


def create_prompt_renderer(
    prompt_cfg: Optional[PromptTemplateConfig] = None,
    *,
    base_dir: Optional[Path] = None,
) -> PromptRenderer:
    cfg = prompt_cfg or PromptTemplateConfig()
    encoding = cfg.encoding if cfg.encoding else "utf-8"
    root = base_dir or cfg.get_config_root()

    system_text = _load_template_text(
        cfg=cfg,
        template_value=getattr(cfg, "system_template", None),
        template_path=getattr(cfg, "system_template_path", None),
        base_dir=root,
        encoding=encoding,
        label="system",
    )
    user_text = _load_template_text(
        cfg=cfg,
        template_value=getattr(cfg, "user_template", None),
        template_path=getattr(cfg, "user_template_path", None),
        base_dir=root,
        encoding=encoding,
        label="user",
    )

    renderer = PromptRenderer.from_strings(system_text, user_text)
    logger.debug(
        "Created prompt renderer using templates (system=%d chars, user=%d chars)",
        len(system_text),
        len(user_text),
    )
    return renderer


__all__ = ["PromptRenderer", "create_prompt_renderer"]
