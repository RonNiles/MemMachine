from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.semantic_memory.semantic_llm import (
    _features_to_consolidation_format,
    _features_to_llm_format,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine_server.semantic_memory.semantic_model import (
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
)


@pytest.fixture
def magic_mock_llm_model() -> MagicMock:
    mock = MagicMock(spec=LanguageModel)
    mock.generate_parsed_response = AsyncMock()
    return mock


@pytest.fixture
def basic_features():
    return [
        SemanticFeature(
            category="Profile",
            tag="food",
            feature_name="favorite_pizza",
            value="peperoni pizza",
        ),
        SemanticFeature(
            category="Profile",
            tag="food",
            feature_name="favorite_bread",
            value="whole grain",
        ),
    ]


@pytest.mark.asyncio
async def test_empty_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    # Given an empty LLM response from the prompt
    magic_mock_llm_model.generate_parsed_response.return_value = {"commands": []}

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect no commands to be returned
    assert commands == []


@pytest.mark.asyncio
async def test_single_command_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    # Given a single LLM response from the prompt
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert commands == [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            tag="car",
            feature="favorite_car_color",
            value="blue",
        ),
    ]


@pytest.mark.asyncio
async def test_multiple_commands_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue",
            },
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car",
                "value": "Tesla",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue Tesla cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert len(commands) == 2
    assert commands[0].command == SemanticCommandType.ADD
    assert commands[0].feature == "favorite_car_color"
    assert commands[1].command == SemanticCommandType.ADD
    assert commands[1].feature == "favorite_car"


@pytest.mark.asyncio
async def test_empty_consolidate_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "consolidated_memories": [],
        "keep_memories": None,
    }

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.consolidated_memories == []
    assert new_feature_resp.keep_memories is None


@pytest.mark.asyncio
async def test_no_action_consolidate_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "keep_memories": [],
        "consolidated_memories": [],
    }

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.keep_memories == []
    assert new_feature_resp.consolidated_memories == []


@pytest.mark.asyncio
async def test_consolidate_with_valid_memories(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "keep_memories": [1, 2],
        "consolidated_memories": [
            {
                "tag": "food",
                "feature": "favorite_pizza",
                "value": "pepperoni",
            },
            {
                "tag": "food",
                "feature": "favorite_drink",
                "value": "water",
            },
        ],
    }

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert result is not None
    assert result.keep_memories == ["1", "2"]
    assert len(result.consolidated_memories) == 2
    assert result.consolidated_memories[0].feature == "favorite_pizza"
    assert result.consolidated_memories[1].feature == "favorite_drink"


@pytest.mark.asyncio
async def test_llm_feature_update_handles_model_api_error(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    from memmachine_server.common.data_types import ExternalServiceAPIError

    # Given an LLM that raises API error
    magic_mock_llm_model.generate_parsed_response.side_effect = ExternalServiceAPIError(
        "API timeout",
    )

    with pytest.raises(ExternalServiceAPIError):
        await llm_feature_update(
            features=basic_features,
            message_content="I like blue cars",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )


@pytest.mark.asyncio
async def test_llm_feature_update_with_delete_command(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "delete",
                "tag": "food",
                "feature": "favorite_pizza",
                "value": "",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I don't like pizza anymore",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert len(commands) == 1
    assert commands[0].command == SemanticCommandType.DELETE
    assert commands[0].feature == "favorite_pizza"


class TestConsolidationSerialization:
    """Consolidation needs a separate serializer that includes feature IDs
    so the LLM can reference them in ``keep_memories``."""

    @pytest.fixture
    def features_with_ids(self):
        return [
            SemanticFeature(
                category="CodeKnowledge",
                tag="bugfix",
                feature_name="observer_fix",
                value="Fixed observer subagent bug",
                metadata=SemanticFeature.Metadata(id="42"),
            ),
            SemanticFeature(
                category="CodeKnowledge",
                tag="progress",
                feature_name="more_agents",
                value="User added more agents",
                metadata=SemanticFeature.Metadata(id="43"),
            ),
        ]

    def test_update_format_omits_ids(self, features_with_ids):
        """The update serializer intentionally omits IDs — updates don't
        need them."""
        import json

        formatted = _features_to_llm_format(features_with_ids)
        serialized = json.dumps(formatted)

        assert "42" not in serialized
        assert "43" not in serialized

    def test_consolidation_format_includes_ids(self, features_with_ids):
        """The consolidation serializer must include ``metadata.id`` so
        the LLM can return them in ``keep_memories``."""
        import json

        formatted = _features_to_consolidation_format(features_with_ids)
        serialized = json.dumps(formatted)

        assert "42" in serialized
        assert "43" in serialized
        assert "metadata" in serialized

    def test_consolidation_format_preserves_all_fields(self, features_with_ids):
        """Each entry in the consolidation format should have tag, feature,
        value, and metadata.id."""
        formatted = _features_to_consolidation_format(features_with_ids)

        assert len(formatted) == 2
        entry = formatted[0]
        assert entry["tag"] == "bugfix"
        assert entry["feature"] == "observer_fix"
        assert entry["value"] == "Fixed observer subagent bug"
        assert entry["metadata"] == {"id": "42"}


class TestNonAsciiPromptSerialization:
    """Both ``llm_feature_update`` and ``llm_consolidate_features`` embed
    the existing feature set into the user prompt via
    ``json.dumps(..., ensure_ascii=False)``. The non-ASCII payload must
    survive into the prompt as literal Unicode (so the LLM sees
    ``"寿司"`` and not ``"\\u5bff\\u53f8"``) and the prompt must remain
    a valid UTF-8 string."""

    @pytest.fixture
    def non_ascii_features(self):
        return [
            SemanticFeature(
                category="Profile",
                tag="食べ物",  # tag itself is non-ASCII
                feature_name="favorite_dish",
                value="寿司 🍣",
                metadata=SemanticFeature.Metadata(id="100"),
            ),
            SemanticFeature(
                category="Profile",
                tag="préférences",
                feature_name="café",
                value="naïve résumé — Привет",
                metadata=SemanticFeature.Metadata(id="101"),
            ),
        ]

    @pytest.mark.asyncio
    async def test_feature_update_prompt_preserves_non_ascii_literally(
        self,
        magic_mock_llm_model: MagicMock,
        non_ascii_features: list[SemanticFeature],
    ):
        magic_mock_llm_model.generate_parsed_response.return_value = {"commands": []}

        await llm_feature_update(
            features=non_ascii_features,
            message_content="I had 寿司 for lunch",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )

        # The user_prompt is the second positional or 'user_prompt' kwarg.
        call_kwargs = magic_mock_llm_model.generate_parsed_response.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]

        # Literal Unicode reaches the LLM, no escape sequences.
        assert "食べ物" in user_prompt
        assert "寿司 🍣" in user_prompt
        assert "préférences" in user_prompt
        assert "café" in user_prompt
        assert "naïve résumé — Привет" in user_prompt
        assert "\\u" not in user_prompt

        # The prompt is UTF-8 transport-safe.
        assert user_prompt.encode("utf-8").decode("utf-8") == user_prompt

    @pytest.mark.asyncio
    async def test_consolidate_prompt_preserves_non_ascii_literally(
        self,
        magic_mock_llm_model: MagicMock,
        non_ascii_features: list[SemanticFeature],
    ):
        magic_mock_llm_model.generate_parsed_response.return_value = {
            "consolidated_memories": [],
            "keep_memories": None,
        }

        await llm_consolidate_features(
            features=non_ascii_features,
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate features",
        )

        call_kwargs = magic_mock_llm_model.generate_parsed_response.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]

        assert "食べ物" in user_prompt
        assert "寿司 🍣" in user_prompt
        assert "préférences" in user_prompt
        assert "naïve résumé — Привет" in user_prompt
        assert "\\u" not in user_prompt
        assert user_prompt.encode("utf-8").decode("utf-8") == user_prompt

        # The consolidation prompt is bare JSON — verify it still parses
        # and round-trips losslessly.
        import json

        parsed = json.loads(user_prompt)
        assert parsed[0]["tag"] == "食べ物"
        assert parsed[0]["value"] == "寿司 🍣"
        assert parsed[0]["metadata"] == {"id": "100"}
        assert parsed[1]["feature"] == "café"
        assert parsed[1]["value"] == "naïve résumé — Привет"

    @pytest.mark.asyncio
    async def test_feature_update_prompt_old_profile_block_is_valid_json(
        self,
        magic_mock_llm_model: MagicMock,
        non_ascii_features: list[SemanticFeature],
    ):
        """The feature-update prompt wraps the JSON inside ``<OLD_PROFILE>``
        delimiters; the inner block must still parse as JSON so the LLM
        is shown structurally valid input."""
        magic_mock_llm_model.generate_parsed_response.return_value = {"commands": []}

        await llm_feature_update(
            features=non_ascii_features,
            message_content="…",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )

        user_prompt = magic_mock_llm_model.generate_parsed_response.call_args.kwargs[
            "user_prompt"
        ]

        start = user_prompt.index("<OLD_PROFILE>\n") + len("<OLD_PROFILE>\n")
        end = user_prompt.index("\n</OLD_PROFILE>")
        old_profile_json = user_prompt[start:end]

        import json

        parsed = json.loads(old_profile_json)
        assert parsed == {
            "食べ物": {"favorite_dish": "寿司 🍣"},
            "préférences": {"café": "naïve résumé — Привет"},
        }


class TestDeterministicIngestionPromptStability:
    """In deterministic mode the update/consolidation prompts must depend only
    on feature *content* (tag, feature, value) — never on DB insertion order or
    ephemeral row ids. Otherwise the LLM cache can't hit across DB-cleared
    reruns, which is the whole point of ``deterministic_ingestion``."""

    # Same three features every "run"; the caller supplies the run-specific
    # ids and ordering. Sorted by (tag, feature_name, value) the deterministic
    # order is: drink/favorite_drink, food/favorite_bread, food/favorite_pizza.
    _DEFS: ClassVar[list[tuple[str, str, str, str]]] = [
        ("Profile", "food", "favorite_pizza", "pepperoni"),
        ("Profile", "food", "favorite_bread", "whole grain"),
        ("Profile", "drink", "favorite_drink", "water"),
    ]

    @classmethod
    def _features(cls, ids: list[str]) -> list[SemanticFeature]:
        return [
            SemanticFeature(
                category=c,
                tag=t,
                feature_name=fn,
                value=v,
                metadata=SemanticFeature.Metadata(id=i),
            )
            for (c, t, fn, v), i in zip(cls._DEFS, ids, strict=True)
        ]

    def test_old_profile_render_is_order_independent(self):
        import json

        run_a = self._features(["100", "200", "300"])
        run_b = list(reversed(self._features(["900", "910", "920"])))

        a = json.dumps(
            _features_to_llm_format(run_a, deterministic=True), ensure_ascii=False
        )
        b = json.dumps(
            _features_to_llm_format(run_b, deterministic=True), ensure_ascii=False
        )

        # Byte-identical despite different input order...
        assert a == b
        # ...and the update prompt never leaks DB ids.
        for db_id in ("100", "200", "300", "900", "910", "920"):
            assert db_id not in a

    def test_consolidation_render_uses_positional_ids(self):
        run = self._features(["100", "200", "300"])
        ordered = sorted(run, key=lambda f: (f.tag, f.feature_name, f.value))

        formatted = _features_to_consolidation_format(ordered, deterministic=True)

        # Ids are positions (0..N-1), not the real DB row ids.
        assert [e["metadata"]["id"] for e in formatted] == ["0", "1", "2"]
        # Content is still present and correctly ordered.
        assert formatted[0]["feature"] == "favorite_drink"
        assert formatted[2]["feature"] == "favorite_pizza"

    @pytest.mark.asyncio
    async def test_consolidation_prompt_identical_across_db_cleared_runs(
        self, magic_mock_llm_model: MagicMock
    ):
        magic_mock_llm_model.generate_parsed_response.return_value = {
            "keep_memories": [],
            "consolidated_memories": [],
        }

        await llm_consolidate_features(
            features=self._features(["100", "200", "300"]),
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
            deterministic=True,
        )
        prompt_a = magic_mock_llm_model.generate_parsed_response.call_args.kwargs[
            "user_prompt"
        ]

        # Second "run": features inserted in a different order with different
        # DB ids (as happens after a DB wipe + concurrent re-adds).
        await llm_consolidate_features(
            features=list(reversed(self._features(["900", "910", "920"]))),
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
            deterministic=True,
        )
        prompt_b = magic_mock_llm_model.generate_parsed_response.call_args.kwargs[
            "user_prompt"
        ]

        assert prompt_a == prompt_b

    @pytest.mark.asyncio
    async def test_consolidation_maps_positional_keep_ids_back_to_real_ids(
        self, magic_mock_llm_model: MagicMock
    ):
        # The (cached) LLM response keeps position 0, which in deterministic
        # sorted order is the drink/favorite_drink feature.
        magic_mock_llm_model.generate_parsed_response.return_value = {
            "keep_memories": ["0"],
            "consolidated_memories": [],
        }

        result_a = await llm_consolidate_features(
            features=self._features(["100", "200", "300"]),
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
            deterministic=True,
        )
        # drink/favorite_drink had id "300" in run A.
        assert result_a is not None
        assert result_a.keep_memories == ["300"]

        # Same position, different run-specific ids -> resolves to that run's
        # id for the same logical feature ("920").
        result_b = await llm_consolidate_features(
            features=list(reversed(self._features(["900", "910", "920"]))),
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
            deterministic=True,
        )
        assert result_b is not None
        assert result_b.keep_memories == ["920"]

    @pytest.mark.asyncio
    async def test_consolidation_drops_out_of_range_positions(
        self, magic_mock_llm_model: MagicMock
    ):
        magic_mock_llm_model.generate_parsed_response.return_value = {
            "keep_memories": ["0", "99", "not-an-int"],
            "consolidated_memories": [],
        }

        result = await llm_consolidate_features(
            features=self._features(["100", "200", "300"]),
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
            deterministic=True,
        )
        # Only the valid position 0 -> "300" survives.
        assert result is not None
        assert result.keep_memories == ["300"]

    @pytest.mark.asyncio
    async def test_default_mode_unchanged_keeps_real_ids_and_db_order(
        self, magic_mock_llm_model: MagicMock
    ):
        magic_mock_llm_model.generate_parsed_response.return_value = {
            "keep_memories": ["200"],
            "consolidated_memories": [],
        }

        run = self._features(["100", "200", "300"])
        result = await llm_consolidate_features(
            features=run,
            model=magic_mock_llm_model,
            consolidate_prompt="Consolidate",
        )
        # Without deterministic mode the prompt carries real ids and
        # keep_memories passes through untranslated.
        prompt = magic_mock_llm_model.generate_parsed_response.call_args.kwargs[
            "user_prompt"
        ]
        assert '"id": "100"' in prompt
        assert result is not None
        assert result.keep_memories == ["200"]
