"""LLM helpers for extracting and consolidating semantic features."""

import json
import logging

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    TypeAdapter,
    field_validator,
    validate_call,
)

from memmachine_server.common.episode_store import EpisodeIdT
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.semantic_memory.semantic_model import (
    SemanticCommand,
    SemanticFeature,
)

logger = logging.getLogger(__name__)


def _content_sort_key(feature: SemanticFeature) -> tuple[str, str, str]:
    """Stable, DB-id-independent ordering key for a feature."""
    return (feature.tag, feature.feature_name, feature.value)


def _features_to_llm_format(
    features: list[SemanticFeature],
    *,
    deterministic: bool = False,
) -> dict[str, dict[str, str]]:
    # Sort by content so the rendered <OLD_PROFILE> depends only on which
    # features exist, not on their DB insertion order / created_at — which is
    # not reproducible across DB-cleared reruns. Required for LLM cache hits.
    if deterministic:
        features = sorted(features, key=_content_sort_key)

    structured_features: dict[str, dict[str, str]] = {}

    for feature in features:
        structured_features.setdefault(feature.tag, {})[feature.feature_name] = (
            feature.value
        )

    return structured_features


def _features_to_consolidation_format(
    features: list[SemanticFeature],
    *,
    deterministic: bool = False,
) -> list[dict[str, str | dict[str, str | None]]]:
    """Serialize features for the consolidation LLM.

    Each memory is a flat object with ``tag``, ``feature``, ``value`` and a
    ``metadata.id`` the LLM can reference in ``keep_memories``.

    In deterministic mode the id is the feature's **position** in the given
    list (``0..N-1``) rather than its ephemeral DB row id, so the prompt — and
    thus the cache key — is identical across DB-cleared reruns. The caller must
    pre-order ``features`` deterministically and map the returned positions back
    to real ids (see ``_resolve_positional_keep_ids``).
    """
    return [
        {
            "tag": f.tag,
            "feature": f.feature_name,
            "value": f.value,
            "metadata": {"id": str(i) if deterministic else f.metadata.id},
        }
        for i, f in enumerate(features)
    ]


class _SemanticFeatureUpdateRes(BaseModel):
    """Schema used to validate parsed feature-update commands returned by the LLM."""

    commands: list[SemanticCommand] = Field(default_factory=list)


@validate_call
async def llm_feature_update(
    features: list[SemanticFeature],
    message_content: str,
    model: InstanceOf[LanguageModel],
    update_prompt: str,
    *,
    deterministic: bool = False,
) -> list[SemanticCommand]:
    """Generate feature update commands from an incoming message using the LLM."""
    user_prompt = (
        "The old feature set is provided below:\n"
        "<OLD_PROFILE>\n"
        f"{json.dumps(_features_to_llm_format(features, deterministic=deterministic), ensure_ascii=False)}\n"
        "</OLD_PROFILE>\n"
        "\n"
        "The history is provided below:\n"
        "<HISTORY>\n"
        f"{message_content}\n"
        "</HISTORY>\n"
    )

    parsed_output = await model.generate_parsed_response(
        system_prompt=update_prompt,
        user_prompt=user_prompt,
        output_format=_SemanticFeatureUpdateRes,
    )

    if parsed_output is None:
        return []

    validated_output = TypeAdapter(_SemanticFeatureUpdateRes).validate_python(
        parsed_output,
    )
    return validated_output.commands


class LLMReducedFeature(BaseModel):
    """Minimal feature payload emitted by the consolidation prompt for reinsertion."""

    tag: str
    feature: str
    value: str

    @field_validator("tag", "feature", "value", mode="after")
    @classmethod
    def strip_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            return v.replace("\x00", "")
        return v


class SemanticConsolidateMemoryRes(BaseModel):
    """LLM response describing merged features and ids of features to retain."""

    consolidated_memories: list[LLMReducedFeature] = Field(default_factory=list)
    keep_memories: list[EpisodeIdT] | None
    model_config = ConfigDict(coerce_numbers_to_str=True)


def _resolve_positional_keep_ids(
    ordered_features: list[SemanticFeature],
    keep_positions: list[EpisodeIdT],
) -> list[EpisodeIdT]:
    """Map positional keep ids from the consolidation LLM back to real ids.

    In deterministic mode the consolidation prompt presents each feature with
    its position (``0..N-1``) as the id, so the request — and thus the cache
    key — doesn't depend on ephemeral DB row ids. The LLM therefore returns
    positions in ``keep_memories``; translate them back to the real feature
    ids in the same order used to build the prompt. Out-of-range or
    non-integer entries are dropped.
    """
    resolved: list[EpisodeIdT] = []
    for pos in keep_positions:
        try:
            idx = int(pos)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(ordered_features):
            real_id = ordered_features[idx].metadata.id
            if real_id is not None:
                resolved.append(real_id)
    return resolved


@validate_call
async def llm_consolidate_features(
    features: list[SemanticFeature],
    model: InstanceOf[LanguageModel],
    consolidate_prompt: str,
    *,
    deterministic: bool = False,
) -> SemanticConsolidateMemoryRes | None:
    """Merge overlapping features and return consolidation commands from the LLM."""
    # In deterministic mode order by content once: the same order is used both
    # to assign positional ids in the prompt and to map the returned positions
    # back to real feature ids.
    ordered_features = (
        sorted(features, key=_content_sort_key) if deterministic else features
    )
    parsed_output = await model.generate_parsed_response(
        system_prompt=consolidate_prompt,
        user_prompt=json.dumps(
            _features_to_consolidation_format(
                ordered_features, deterministic=deterministic
            ),
            ensure_ascii=False,
        ),
        output_format=SemanticConsolidateMemoryRes,
    )

    if parsed_output is None:
        return None

    validated_output = TypeAdapter(SemanticConsolidateMemoryRes).validate_python(
        parsed_output,
    )

    if deterministic and validated_output.keep_memories is not None:
        validated_output.keep_memories = _resolve_positional_keep_ids(
            ordered_features, validated_output.keep_memories
        )

    return validated_output
