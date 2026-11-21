from __future__ import annotations

import json
import sys
import pytest
from types import ModuleType, SimpleNamespace

from ohpu.benchmarks import pm3_native
from ohpu.llm import LLM, Usage
from ohpu.experiments import pm3_proxy
from ohpu.per_user_adaptation.core.models import make_minimal_config
from ohpu.per_user_adaptation.executor import ExecutableHarness
from ohpu.per_user_adaptation.program import HarnessProgram


@pytest.mark.parametrize("parsed", [[2, 0, 1], [], "ranked_indices", 3, True, None])
def test_e5_response_guard_uses_official_missing_ranking_fallback(parsed):
    module = SimpleNamespace(extract_json_from_response=lambda raw: parsed)
    pm3_native._install_e5_response_type_guard(module)
    assert module.extract_json_from_response("original answer") == {}


def test_e5_response_guard_preserves_objects_and_is_idempotent():
    response = {"ranked_indices": [2, 0, 1], "stopped_due_to": "expired"}
    calls = []

    def parser(raw):
        calls.append(raw)
        return response

    module = SimpleNamespace(extract_json_from_response=parser)
    pm3_native._install_e5_response_type_guard(module)
    installed = module.extract_json_from_response
    pm3_native._install_e5_response_type_guard(module)
    assert module.extract_json_from_response is installed
    assert module.extract_json_from_response("original answer") is response
    assert calls == ["original answer"]


def test_new_suggestions_list_response_keeps_raw_answer_for_judge(monkeypatch):
    module = pm3_proxy.official_module("evaluation.tasks.new_suggestions")
    raw = '[{"recommendation": "Try something new", "hashtags": ["fresh"]}]'
    monkeypatch.setattr(module, "extract_json_from_response", lambda text: [{"recommendation": "Try something new"}]
        if text == raw else {"alignment_score": 2.5, "reasoning": "supported", "hard_fail": False})
    with pytest.raises(AttributeError, match="get"):
        module.compute_new_suggestions_chatbot_metrics(module.extract_json_from_response(raw), raw, {}, None, False)
    pm3_native._install_e5_response_type_guard(module)
    seen = []
    monkeypatch.setattr(module.prompts_mod, "judge_new_suggestions_chatbot_prompt",
        lambda **kwargs: seen.append(kwargs["agent_response"]) or "judge this unchanged answer")
    result = module.compute_new_suggestions_chatbot_metrics(module.extract_json_from_response(raw), raw, {},
        lambda prompt: "judge result", True)
    assert seen == [raw]
    assert result["alignment_score"] == 2.5 and result["valid"]


def test_calendar_state_uses_paper_strict_before_timestamp_boundary() -> None:
    class UpstreamInclusiveBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.modifications = [
                {"ts": 99, "entry_id": "past"},
                {"ts": 100, "entry_id": "same-time"},
            ]

        def get_calendar_state(self, user_id: str, as_of_timestamp: int) -> dict:
            assert user_id == "u"
            self.calls.append(as_of_timestamp)
            return {
                row["entry_id"]: row
                for row in self.modifications
                if row["ts"] <= as_of_timestamp
            }

    strict_type = pm3_native._paper_strict_backend_type(
        UpstreamInclusiveBackend
    )
    backend = strict_type()

    assert set(backend.get_calendar_state("u", 100)) == {"past"}
    assert backend.calls == [99]
    assert pm3_native.CALENDAR_HISTORY_CUTOFF.endswith("< T_test")


def test_official_repetition_response_cache_is_isolated_per_arm() -> None:
    module = ModuleType("fake_over_personalization")
    module._C1C_CLUSTER_CACHE = {}
    caches = pm3_native._ConditionClusterCaches(module)
    instance = {"cluster_id": "u-cluster-1"}

    b0 = caches.activate(
        "B0",
        "over_personalization_repetition_recsys",
        instance,
        query_id="q0",
        t_test=100,
        harness_checkpoint=0,
    )
    assert b0["hit_before_dispatch"] is False
    module._C1C_CLUSTER_CACHE["u-cluster-1"] = {"responses": ["b0"]}
    assert caches.finish(b0)["origin"]["harness_checkpoint"] == "W0"

    ours = caches.activate(
        "Ours",
        "over_personalization_repetition_recsys",
        instance,
        query_id="q0",
        t_test=100,
        harness_checkpoint=0,
    )
    assert ours["hit_before_dispatch"] is False
    assert module._C1C_CLUSTER_CACHE == {}
    module._C1C_CLUSTER_CACHE["u-cluster-1"] = {"responses": ["ours"]}
    caches.finish(ours)

    b0_later = caches.activate(
        "B0",
        "over_personalization_repetition_recsys",
        instance,
        query_id="q1",
        t_test=200,
        harness_checkpoint=2,
    )
    assert b0_later["hit_before_dispatch"] is True
    assert module._C1C_CLUSTER_CACHE["u-cluster-1"]["responses"] == ["b0"]
    audit = caches.finish(b0_later)
    assert audit["hit"] is True
    assert audit["origin"] == {
        "query_id": "q0",
        "t_test": 100,
        "harness_checkpoint": "W0",
    }


def test_prompt_builder_profile_firewall_preserves_causal_backend_only() -> None:
    ground_truth = ModuleType("fake_ground_truth_builders")
    proactive = ModuleType("fake_proactive_actions")

    def build_for_task(task_id, backend, user_id, t_test, instance):
        del task_id, instance
        return repr({
            "profile": backend.get_full_profile(user_id),
            "friend": backend.get_friend(user_id, "friend-1"),
            "events": backend.get_events(user_id, "instagram", t_test),
        })

    ground_truth.build_for_task = build_for_task
    proactive._build_user_state_summary = lambda backend, user_id: repr(
        backend.get_full_profile(user_id)
    )

    class Backend:
        def get_full_profile(self, user_id):
            return {"user_id": user_id, "hidden_personas": ["PRIVATE"]}

        def get_friend(self, user_id, friend_id):
            return {"user_id": user_id, "friend_id": friend_id}

        def get_events(self, user_id, app, t_test):
            return [{"user_id": user_id, "app": app, "ts": t_test - 1}]

    pm3_native._install_paper_prompt_firewall(ground_truth, proactive)
    rendered = ground_truth.build_for_task(
        "task", Backend(), "u", 100, {}
    )

    assert "PRIVATE" not in rendered
    assert "friend-1" not in rendered
    assert "instagram" in rendered
    assert proactive._build_user_state_summary(Backend(), "u") == (
        "(profile withheld; infer user state from causal history)"
    )


def test_agent_prompt_firewall_fails_closed_on_private_label_field() -> None:
    clean = pm3_native._agent_prompt_audit(
        "causal history followed by the public user query"
    )
    assert clean["forbidden_markers_exposed"] == []

    try:
        pm3_native._agent_prompt_audit(
            'answer this query; "held_out_idx": 7'
        )
    except pm3_native.AgentPromptLeakError as exc:
        assert "held_out_idx" in str(exc)
    else:
        raise AssertionError("evaluator-private prompt field must fail closed")


def test_pm3_code_hook_receives_only_the_audited_public_user_request() -> None:
    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return {
                "content": "done",
                "tool_calls": [],
                "usage": {"prompt": 1, "completion": 1, "reasoning": 0},
            }

    program = HarnessProgram('''\
PROGRAM_API_VERSION = 1
def remember(context):
    return {}
def recall(context):
    return {}
def apply(context):
    if context.get("stage") != "generation":
        return {}
    query = context.get("query", "")
    if "INTERNAL-BACKEND-MARKER" in query:
        return {"prefix": "PRIVATE-SEEN"}
    if "PUBLIC-REQUEST" in query:
        return {"prefix": "PUBLIC-SEEN"}
    return {}
def guard(context):
    return {}
''')
    llm = RecordingLLM()
    adapter = pm3_native.HarnessQueryLLM(
        llm,
        ExecutableHarness(
            user_id="u", config=make_minimal_config(), program=program
        ),
        query_id="q",
        task_type="preference",
        condition="test",
    )

    assert adapter.query_llm([
        {"role": "system", "content": "INTERNAL-BACKEND-MARKER"},
        {"role": "user", "content": "PUBLIC-REQUEST"},
    ], use_history=True) == "done"

    system = llm.calls[0]["messages"][0]["content"]
    assert "PUBLIC-SEEN" in system
    assert "PRIVATE-SEEN" not in system


def test_online_task_request_removes_history_and_keeps_current_task() -> None:
    prompt = (
        "# Task: answer the request\n\n"
        "## Cross-app user history (time-masked)\n"
        + "RAW_EVENT_SHOULD_NOT_REACH_HARNESS\n" * 2000
        + "\n## Output\nPUBLIC_CURRENT_REQUEST"
    )
    request, audit = pm3_native._online_task_request(prompt)

    assert "RAW_EVENT_SHOULD_NOT_REACH_HARNESS" not in request
    assert "PUBLIC_CURRENT_REQUEST" in request
    assert audit["raw_history_forwarded"] is False
    assert audit["history_sections_removed"] == 1
    assert audit["source_prompt_characters"] > audit[
        "current_request_characters"
    ]

    suffixed, suffix_audit = pm3_native._online_task_request(
        "## User history (time-masked) — DO NOT apply here\n"
        "RAW_SUFFIXED_HISTORY\n"
        "## Output\nCURRENT_ONLY"
    )
    assert "RAW_SUFFIXED_HISTORY" not in suffixed
    assert "CURRENT_ONLY" in suffixed
    assert suffix_audit["history_sections_removed"] == 1


def test_online_snapshot_cache_never_reads_backend_history() -> None:
    class ExplodingBackend:
        def __getattr__(self, name):
            raise AssertionError(f"online cache must not call backend.{name}")

    cache = pm3_native.OnlineSnapshotCache("compact preference memory")
    text, stats = cache.get_or_build(
        ExplodingBackend(), "u", 123, "gpt", 800_000
    )

    assert "compact preference memory" in text
    assert pm3_native._online_task_request(text)[0] == ""
    assert stats["online_memory"] is True
    assert stats["raw_history_chars"] == 0
    assert stats["raw_history_serialized"] is False


def test_nested_memory_headings_cannot_bypass_recall_through_native_request() -> None:
    memory = "# User memory\n## Preferences\n- PRIVATE_MEMORY_FACT\n## Recent events\n- OTHER_PRIVATE_FACT"
    cache = pm3_native.OnlineSnapshotCache(memory)
    marked, _ = cache.get_or_build(None, "u", 123, "test", 800_000)
    # Cover both official heading layouts and embedding without a heading.
    for heading in ("## Cross-app user history (time-masked)\n", "## User history (time-masked)\n", ""):
        prompt = "# Task\n" + heading + marked + "\n## Output\nPUBLIC_REQUEST"
        request, audit = pm3_native._online_task_request(prompt)
        assert "PRIVATE" not in request
        assert "PUBLIC_REQUEST" in request and "## Output" in request
        assert audit["delimited_memory_sections_removed"] == 1
        llm = _RecordingNativeBoundaryLLM()
        program = HarnessProgram('PROGRAM_API_VERSION = 1\ndef remember(c): return {}\ndef recall(c): return {"memory_mode":"no_user_memory"}\ndef apply(c): return {}\ndef guard(c): return {}\n')
        adapter = pm3_native.HarnessQueryLLM(llm, ExecutableHarness("u", make_minimal_config(), memory_md=memory, program=program),
            query_id="q", task_type="preference", condition="test")
        adapter.query_llm(prompt)
        assert "PRIVATE_MEMORY_FACT" not in repr(llm.calls)
        assert "OTHER_PRIVATE_FACT" not in repr(llm.calls)
        assert "PUBLIC_REQUEST" in repr(llm.calls)


class _RecordingNativeBoundaryLLM:
    def __init__(self):
        self.calls = []
    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return {"content": "done", "tool_calls": [], "usage": {"prompt": 1, "completion": 1, "reasoning": 0}}


def test_harness_adapter_forwards_only_bounded_request_and_bundle_memory() -> None:
    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return {
                "content": "done",
                "tool_calls": [],
                "usage": {"prompt": 1, "completion": 1, "reasoning": 0},
            }

    program = HarnessProgram('''\
PROGRAM_API_VERSION = 1
def remember(context):
    return {}
def recall(context):
    return {}
def apply(context):
    if context.get("stage") != "generation":
        return {}
    query = context.get("query", "")
    return {"prefix": "LEAKED" if "RAW_HISTORY" in query else "SAFE"}
def guard(context):
    return {}
''')
    llm = RecordingLLM()
    adapter = pm3_native.HarnessQueryLLM(
        llm,
        ExecutableHarness(
            user_id="u",
            config=make_minimal_config(),
            memory_md="compact online memory",
            program=program,
        ),
        query_id="q",
        task_type="preference",
        condition="test",
    )
    prompt = (
        "# Task\n"
        "## Cross-app user history (time-masked)\n"
        + "RAW_HISTORY " * 5000
        + "\n## Output\nPUBLIC_REQUEST"
    )

    assert adapter.query_llm(prompt) == "done"
    trace = adapter.traces[0]
    assert any(
        event.get("hook") == "apply"
        and event.get("stage") == "generation"
        and event.get("effective")
        and event.get("applied_keys") == ["prefix"]
        for event in trace["program_report"]["events"]
    )
    assert all(
        "RAW_HISTORY" not in str(message.get("content"))
        for message in llm.calls[0]["messages"]
    )
    assert "compact online memory" in llm.calls[0]["messages"][0]["content"]
    assert adapter.prompt_audits[0]["raw_history_forwarded"] is False


def test_online_task_request_fails_closed_on_unbounded_nonhistory_prompt() -> None:
    try:
        pm3_native._online_task_request("PUBLIC " * 40_000)
    except pm3_native.HarnessInputBoundaryError as exc:
        assert "online harness bound" in str(exc)
    else:
        raise AssertionError("unbounded current request must fail closed")


def test_summary_separates_released_rows_from_official_scored_rows(
    monkeypatch,
) -> None:
    registry = ModuleType("evaluation.task_registry")
    registry.normalize_task_type = lambda name: name
    registry.DROPPED_TASK_TYPES = {"retired_task"}
    registry.TASK_TYPE_META = {"active_task": {}, "retired_task": {}}
    monkeypatch.setitem(sys.modules, "evaluation.task_registry", registry)

    class Aggregate:
        @staticmethod
        def _e6_paired_f1(rows):
            del rows
            return {}

        @staticmethod
        def _build_token_accuracy_table(rows, e6):
            del rows, e6
            return [
                {"task_type": "ALL (micro, row-weighted)", "accuracy_pct": 50.0},
                {
                    "task_type": "ALL-JUDGE (micro, judge tasks only)",
                    "accuracy_pct": 40.0,
                },
            ]

    class RunEval:
        @staticmethod
        def _summarize_by_task(rows):
            return {"rows_received": len(rows)}

    runtime = pm3_native.OfficialPM3Runtime.__new__(
        pm3_native.OfficialPM3Runtime
    )
    runtime.revision = "test-revision"
    runtime.aggregate = Aggregate()
    runtime.run_eval = RunEval()
    records = [
        {
            "task_type": "active_task",
            "status": "ok",
            "metrics_json": json.dumps({"score": 1}),
        },
        {
            "task_type": "retired_task",
            "status": "ok",
            "metrics_json": json.dumps({"score": 1}),
        },
    ]

    summary = runtime.summarize(records)

    assert summary["n_queries_released"] == 2
    assert summary["n_queries_scored"] == 1
    assert summary["n_queries_dropped_by_official_registry"] == 1
    assert summary["released_task_type_count"] == 2
    assert summary["scored_task_type_count"] == 1
    assert summary["dropped_queries_by_task"] == {"retired_task": 1}
    assert summary["by_task"] == {"rows_received": 1}
    assert summary["metric_complete"] is True
    assert summary["lifecycle_pairing"]["complete_pairs"] == 0


def test_repeated_native_headline_applies_hard_rules_before_averaging():
    runtime = pm3_native.OfficialPM3Runtime.__new__(pm3_native.OfficialPM3Runtime)
    runtime.revision = "test-revision"
    runtime.aggregate = pm3_proxy.official_module("scripts.aggregate_eval")
    runtime.run_eval = pm3_proxy.official_module("evaluation.run_eval")
    draws = [{"pr_preference_alignment_score": 8., "pr_privacy_leak_violated": index == 0}
             for index in range(3)]
    records = [{"query_id": "chat", "task_type": "chatbot_personalized_response", "status": "ok",
                "metrics_json": json.dumps(draws[0]), "_judge_repeats": draws},
               {"query_id": "rule", "task_type": "active_mistake_prevention", "status": "ok",
                "metrics_json": '{"correct":1}'}]
    summary = runtime.summarize(records)
    assert summary["n_queries"] == summary["n_ok"] == 2
    assert summary["fixed_answer_judge_repeats"] == 3
    assert summary["judge_repeat_scores"] == [.5, .9, .9]
    assert summary["score_0_1"] == pytest.approx(2.3 / 3)
    assert summary["evaluation_complete"] is True
    assert len(summary["by_task_repeats"]) == 3


def test_native_judge_repeats_preserve_lifecycle_pairing_and_denominator():
    runtime = pm3_native.OfficialPM3Runtime.__new__(pm3_native.OfficialPM3Runtime)
    runtime.revision = "test-revision"
    runtime.aggregate = pm3_proxy.official_module("scripts.aggregate_eval")
    runtime.run_eval = pm3_proxy.official_module("evaluation.run_eval")
    draws = [{"pr_preference_alignment_score": score} for score in (3., 7., 9.)]
    judged = {"query_id": "chat", "task_type": "chatbot_personalized_response", "status": "ok",
              "metrics_json": json.dumps(draws[0])}
    lifecycle = [_lifecycle_side("pre", 2 / 3), _lifecycle_side("post", 0.)]
    separate = [runtime.summarize([{**judged, "metrics_json": json.dumps(metrics)}, *lifecycle])
                for metrics in draws]
    repeated = runtime.summarize([{**judged, "_judge_repeats": draws}, *lifecycle])
    assert repeated["evaluation_complete"] is True
    assert repeated["n_queries"] == 3
    assert repeated["lifecycle_pairing"]["complete_pairs"] == 1
    assert repeated["n_queries_scored"] == separate[0]["n_queries_scored"]
    assert repeated["score_0_1"] == pytest.approx(sum(item["score_0_1"] for item in separate) / 3)
    assert repeated["judge_repeat_scores"] == [item["score_0_1"] for item in separate]
    with pytest.raises(ValueError):
        runtime.summarize([{**judged, "_judge_repeats": draws},
                          {**judged, "query_id": "another", "_judge_repeats": draws[:2]}])


def _lifecycle_summary_runtime(monkeypatch, captured_rows):
    registry = ModuleType("evaluation.task_registry")
    registry.normalize_task_type = lambda name: name
    registry.DROPPED_TASK_TYPES = set()
    registry.TASK_TYPE_META = {pm3_native.LIFECYCLE_TASK_TYPE: {}}
    monkeypatch.setitem(sys.modules, "evaluation.task_registry", registry)

    class Aggregate:
        @staticmethod
        def _e6_paired_f1(rows):
            del rows
            return {}

        @staticmethod
        def _build_token_accuracy_table(rows, e6):
            del e6
            captured_rows.extend(dict(row) for row in rows)
            scores = [
                row["_metrics"]["lifecycle_score"]
                for row in rows
                if "lifecycle_score" in row.get("_metrics", {})
            ]
            accuracy = (
                50.0 * (1.0 + sum(scores) / len(scores))
                if scores else ""
            )
            return [{
                "task_type": "ALL (micro, row-weighted)",
                "accuracy_pct": accuracy,
            }]

    class RunEval:
        @staticmethod
        def _summarize_by_task(rows):
            return {"rows_received": len(rows)}

    runtime = pm3_native.OfficialPM3Runtime.__new__(
        pm3_native.OfficialPM3Runtime
    )
    runtime.revision = "test-revision"
    runtime.aggregate = Aggregate()
    runtime.run_eval = RunEval()
    return runtime


def _lifecycle_side(phase: str, match_rate: float) -> dict:
    return {
        "query_id": f"u:0001:e5_same-canonical_{phase}",
        "user_id": "u",
        "task_type": pm3_native.LIFECYCLE_TASK_TYPE,
        "ts": "2" if phase == "post" else "1",
        "status": "ok",
        "metrics_json": json.dumps({
            "match_rate_at_3": match_rate,
            "matching_in_top1": int(match_rate > 0),
        }),
    }


def test_summary_synthesizes_official_lifecycle_score_from_complete_pair(
    monkeypatch,
) -> None:
    captured = []
    runtime = _lifecycle_summary_runtime(monkeypatch, captured)

    summary = runtime.summarize([
        _lifecycle_side("pre", 2 / 3),
        _lifecycle_side("post", 1 / 3),
    ])

    score_rows = [
        row for row in captured
        if "lifecycle_score" in row.get("_metrics", {})
    ]
    assert len(score_rows) == 1
    assert score_rows[0]["_metrics"] == {
        "pre.match_rate_at_3": 2 / 3,
        "post.match_rate_at_3": 1 / 3,
        "lifecycle_score": pytest.approx(1 / 3),
        "post.hard_violation_at_1": 1.0,
    }
    assert summary["n_queries"] == 2
    assert summary["by_task"] == {"rows_received": 3}
    assert summary["lifecycle_pairing"] == {
        "lifecycle_side_rows": 2,
        "complete_pairs": 1,
        "synthesized_pair_rows": 1,
        "existing_lifecycle_score_rows": 0,
        "unpaired_side_count": 0,
        "unpaired_sides": [],
        "metric_complete": True,
    }
    assert summary["metric_complete"] is True
    assert summary["evaluation_complete"] is True
    assert summary["score_0_1"] == pytest.approx(2 / 3)


def test_summary_does_not_duplicate_existing_lifecycle_score(
    monkeypatch,
) -> None:
    captured = []
    runtime = _lifecycle_summary_runtime(monkeypatch, captured)
    pair = {
        "query_id": "u:0003:e5_pair_same-canonical",
        "user_id": "u",
        "task_type": pm3_native.LIFECYCLE_TASK_TYPE,
        "ts": "2",
        "status": "ok",
        "metrics_json": json.dumps({"lifecycle_score": 1 / 3}),
    }

    summary = runtime.summarize([
        _lifecycle_side("pre", 2 / 3),
        _lifecycle_side("post", 1 / 3),
        pair,
    ])

    assert sum(
        "lifecycle_score" in row.get("_metrics", {}) for row in captured
    ) == 1
    pairing = summary["lifecycle_pairing"]
    assert pairing["complete_pairs"] == 1
    assert pairing["synthesized_pair_rows"] == 0
    assert pairing["existing_lifecycle_score_rows"] == 1
    assert pairing["metric_complete"] is True


def test_summary_fails_closed_for_unpaired_lifecycle_side(
    monkeypatch,
) -> None:
    captured = []
    runtime = _lifecycle_summary_runtime(monkeypatch, captured)

    summary = runtime.summarize([_lifecycle_side("pre", 2 / 3)])

    assert summary["metric_complete"] is False
    assert summary["evaluation_complete"] is False
    assert summary["score_0_1"] is None
    assert "lifecycle metric is incomplete" in summary["invalid_reason"]
    pairing = summary["lifecycle_pairing"]
    assert pairing["complete_pairs"] == 0
    assert pairing["unpaired_side_count"] == 1
    assert pairing["unpaired_sides"] == [{
        "query_id": "u:0001:e5_same-canonical_pre",
        "phase": "pre",
        "reason": "missing_other_side",
    }]


def test_private_judge_matches_the_official_request_shape() -> None:
    """The pinned evaluator sends only model and messages for its judge.

    A caller-chosen ``max_completion_tokens`` is not neutral: reasoning tokens
    come out of it, an exhausted budget returns empty content, and
    ``personalization_rubric.score`` turns an unparseable reply into 0.0 with
    no error -- a judge outage that reads as a terrible answer.
    """

    sent: list[dict] = []

    class _Completions:
        def create(self, **payload):
            sent.append(payload)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"over_personalization": 9}', tool_calls=[],
                    ),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
                ),
            )

    llm = LLM(
        model="gpt-5.6-luna",
        reasoning_effort="low",
        usage=Usage(),
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()),
        ),
    )
    judge = pm3_native.PrivateJudgeQueryLLM(llm, model="gpt-5.6-luna")

    assert judge.query_llm("rate this") == '{"over_personalization": 9}'
    assert set(sent[0]) == {"model", "messages", "seed"}
    assert sent[0]["model"] == "gpt-5.6-luna"


def _judge_client(contents):
    """An OpenAI-shaped client that replays `contents`, then repeats the last."""

    calls = []

    class _Completions:
        def create(self, **payload):
            calls.append(payload)
            content = contents[min(len(calls) - 1, len(contents) - 1)]
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=[]),
                    finish_reason="length" if content == "" else "stop",
                )],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    return client, calls


def test_judge_retries_an_unusable_reply_before_giving_up() -> None:
    """An empty completion is a retryable outage, not a verdict."""

    client, calls = _judge_client(["", "", '{"over_personalization": 8}'])
    llm = LLM(model="gpt-5.6-luna", usage=Usage(), _client=client)
    judge = pm3_native.PrivateJudgeQueryLLM(llm, model="gpt-5.6-luna")

    assert judge.query_llm("rate this") == '{"over_personalization": 8}'
    assert len(calls) == 3
    assert judge.unusable_replies == 2
    assert judge.failures == 0
    assert [call["seed"] for call in calls] == [7, 8, 9]
    assert "response_format" not in calls[0]
    assert all(call["response_format"] == {"type": "json_object"} for call in calls[1:])
    assert all(call["messages"][0] == calls[0]["messages"][0] for call in calls[1:])
    assert all("FORMAT REPAIR" in call["messages"][-1]["content"] for call in calls[1:])
    assert all("max_completion_tokens" not in call for call in calls)


def test_judge_format_repair_does_not_repeat_cached_unusable_response(tmp_path) -> None:
    from ohpu.llm import Cache

    client, calls = _judge_client(["", '{"over_personalization": 8}'])
    llm = LLM(model="gpt-5.6-luna", usage=Usage(), _client=client,
              cache=Cache(str(tmp_path / "judge.sqlite")))
    judge = pm3_native.PrivateJudgeQueryLLM(llm, model="gpt-5.6-luna")
    original = [{"role": "user", "content": "Score this fixed answer and return JSON."}]

    result = judge.query_llm(original, use_history=True)

    assert result == '{"over_personalization": 8}'
    assert len(calls) == 2
    assert judge.unusable_replies == 1
    assert original == [{"role": "user", "content": "Score this fixed answer and return JSON."}]


def test_an_exhausted_judge_raises_and_counts_itself() -> None:
    """Raising alone is not enough -- both official callers swallow it.

    `personalization_rubric.score` records `judge_error` and scores 0.0;
    `llm_metrics.personalization_leak_check` returns a perfect 10. The counter
    is what lets a caller tell an outage from a verdict.
    """

    client, calls = _judge_client([""])
    llm = LLM(model="gpt-5.6-luna", usage=Usage(), _client=client)
    judge = pm3_native.PrivateJudgeQueryLLM(llm, model="gpt-5.6-luna", max_attempts=2)

    with pytest.raises(pm3_native.JudgeUnavailable):
        judge.query_llm("rate this")

    assert len(calls) == 2
    assert judge.failures == 1
