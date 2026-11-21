import json

import pytest

from ohpu.benchmarks import mempal


@pytest.mark.parametrize("candidate,parent_mean,interval,adopted", [
    ({"first": 3, "second": 3}, 0, [-1, 5], False),
    ({"first": 3, "second": 3}, 0, [0, 5], False),
    ({"first": 3, "second": 3}, 0, [1, 5], True),
    ({"first": 3, "second": 3}, 4, [1, 5], False),
    ({"first": 3}, 0, [1, 5], False),
])
def test_single_pass_requires_complete_significant_r0_gain(tmp_path, monkeypatch, candidate, parent_mean, interval, adopted):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "analysis/skill_optimization_diagnosis/mempal_phg.py"
    specification = importlib.util.spec_from_file_location("mempal_single_pass_test", path)
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    users = ["first", "second"]
    plan = {"split": "TRAIN", "test_users": users, "phg": {"replicates": 1,
        "selection_policy": "significant_vs_r0", "arms": {"phg_r0": 0, "phg_r1": 1, "phg_r2": 2}}}
    driver.write(tmp_path / "plan.json", plan)
    driver.write(tmp_path / "rounds/parent.json", {"version": 1, "history": []})
    driver.write(tmp_path / "generator/v2.json", {"outcome": "accepted"})
    panels = {"phg_r0": dict.fromkeys(users, 0), "phg_r1": dict.fromkeys(users, parent_mean), "phg_r2": candidate}
    monkeypatch.setattr(driver.PHG, "arm_scores", lambda out, active, arm: panels[arm])
    monkeypatch.setattr(driver.PHG, "generation_stats", lambda *args: {})
    def paired(scores, reference, name):
        return {"mean_pp": sum(scores[user] - reference[user] for user in users) / len(users), "ci95_pp": interval}
    monkeypatch.setattr(driver.PHG, "paired", paired)
    result = driver.round_summary(tmp_path, 2)
    assert result["parent_decision"]["adopted"] is adopted
    assert driver.read(tmp_path / "rounds/parent.json")["version"] == (2 if adopted else 1)
    assert not (tmp_path / "replicates").exists()


def sample(identifier, day, content):
    return {"sample_id": identifier, "dialogue_timestamp": f"2024-01-{day:02d} 12:00:00",
        "logs": [{"timestamp": f"2024-01-{day:02d} 09:00", "content": "visible activity " + identifier}],
        "dialogue": {"turn_1": {"user": {"action": "hidden action", "content": content},
            "assistant": {"action": "hidden action", "content": "assistant observation"}}},
        "topics": {"topic-1": {"requirement": "public detailed requirement", "implicit_needs": ["HIDDEN_NEED"],
            "candidate_solutions": [{"solution": f"option {index}",
                "feedback": "pos" if index < 2 else "neg" if index < 4 else "neu"} for index in range(8)]}}}


def native():
    return {"history": [sample("history", 1, "history conversation")],
        "query": [sample("first", 2, "FIRST_TARGET_ANSWER"), sample("second", 3, "FUTURE_TARGET_ANSWER")]}


def test_roles_are_reproducible_disjoint_and_complete():
    users = [f"mempal_{index:04d}" for index in range(100)]
    result = mempal.roles(users)
    assert result == mempal.roles(list(reversed(users)))
    assert len(result["train"]) == len(result["test"]) == 50
    assert not set(result["train"]) & set(result["test"])
    assert set(result["train"] + result["test"]) == set(users)
    with pytest.raises(ValueError):
        mempal.roles(users[:-1])


def test_fixed_filter_mask_reports_original_coverage_without_imputation():
    from ohpu.benchmarks.mempal_runtime import summarize_user
    user = {"cases": [{"case_id": "first"}, {"case_id": "third"}],
            "evaluation_mask": {"protocol": "mempal-fixed-content-filter-mask-v1",
                "original_query_ids": ["first", "second", "third"], "excluded_query_ids": ["second"],
                "mask_sha256": "frozen-mask"}}
    rows = [{"query_id": "first", "selection_score": 100, "format_error": None},
            {"query_id": "third", "selection_score": 0, "format_error": None}]
    result = summarize_user(user, rows)
    assert result["evaluation_complete"] and not result["original_evaluation_complete"]
    assert result["accuracy_pct_micro"] == 50
    assert result["original_queries"] == 3 and result["scored_queries"] == 2
    assert result["filtered_queries"] == 1 and result["coverage"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_user(user, rows[:1])


def test_fixed_filter_mask_rejects_unregistered_exclusions():
    from ohpu.benchmarks.mempal_runtime import summarize_user
    user = {"cases": [{"case_id": "first"}], "evaluation_mask": {
        "protocol": "mempal-fixed-content-filter-mask-v1", "original_query_ids": ["first", "second"],
        "excluded_query_ids": ["unknown"], "mask_sha256": "frozen-mask"}}
    with pytest.raises(ValueError, match="Invalid frozen"):
        summarize_user(user, [{"query_id": "first", "selection_score": 100, "format_error": None}])


def test_causal_history_and_label_visibility():
    user = mempal.prepare_user("0000", native())
    first, second = user["cases"]
    construction = json.dumps(user["events"][:user["construction_cutoff"]])
    assert all(text not in construction for text in ("HIDDEN_NEED", "public detailed requirement", "feedback", "hidden action", "FIRST_TARGET_ANSWER"))
    visible = json.dumps(user["events"][:first["history_cutoff"]])
    assert "visible activity first" in visible
    assert "FIRST_TARGET_ANSWER" not in visible and "FUTURE_TARGET_ANSWER" not in visible
    later = json.dumps(user["events"][:second["history_cutoff"]])
    assert "FIRST_TARGET_ANSWER" in later and "FUTURE_TARGET_ANSWER" not in later
    prompt = mempal.request(first)
    assert "public detailed requirement" in prompt and "visible activity first" in prompt
    assert all(text not in prompt for text in ("HIDDEN_NEED", "feedback", "positive", "FIRST_TARGET_ANSWER"))
    assert "visible activity first" not in mempal.request(first, include_logs=False)


@pytest.mark.parametrize("selected,expected", [(["S1", "S2"], 100), (["S1", "S5"], 50),
    (["S1", "S3"], 0), (["S5", "S6"], 0), (["S3", "S5"], -50), (["S3", "S4"], -100)])
def test_official_signed_score(selected, expected):
    case = mempal.prepare_user("0000", native())["cases"][0]
    assert mempal.selection_score(selected, case) == expected


@pytest.mark.parametrize("selected", [["S1"], ["S1", "S1"], ["S1", "S9"], ["S1", 2]])
def test_invalid_selection_rejected(selected):
    with pytest.raises(ValueError):
        mempal.parse_output(json.dumps({"analysis": "reason", "selected_solutions": selected}))


def test_official_visible_late_log_is_audited_and_duplicate_dialogue_rejected():
    raw = native()
    raw["query"][0]["logs"][0]["timestamp"] = "2024-01-04 09:00"
    user = mempal.prepare_user("0000", raw)
    assert user["late_log_timestamps"] == 1
    assert "2024-01-04 09:00" in mempal.request(user["cases"][0])
    raw = native()
    raw["query"][1]["sample_id"] = "first"
    with pytest.raises(ValueError, match="Duplicate"):
        mempal.prepare_user("0000", raw)


def test_construction_digest_contains_only_visible_history():
    user = mempal.prepare_user("0000", native())
    digest = json.loads(mempal.construction_digest(user["events"][:user["construction_cutoff"]]))
    assert digest["total_sessions"] == 1
    assert "FIRST_TARGET_ANSWER" not in json.dumps(digest)


def test_english_source_and_response_language():
    assert mempal.DATASET_FILE.endswith("data_en/input_en.json")
    assert "-en-" in mempal.PROTOCOL
    case = mempal.prepare_user("0000", native())["cases"][0]
    assert "English explanation" in mempal.request(case)


def test_analysis_schema_is_short_and_escape_free_without_changing_labels():
    import jsonschema
    from ohpu.benchmarks.mempal_runtime import RESPONSE_FORMAT
    schema = RESPONSE_FORMAT['json_schema']['schema']
    valid = {'analysis': 'Observed preferences support S1 and S2.', 'selected_solutions': ['S1', 'S2']}
    jsonschema.validate(valid, schema)
    for analysis in ['x' * 241, 'quote " inside', 'backslash \\ repeated', 'line\nbreak', 'tab\tvalue']:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**valid, 'analysis': analysis}, schema)
    assert schema['properties']['selected_solutions']['items']['enum'] == list(mempal.LABELS)
    assert schema['properties']['selected_solutions']['minItems'] == schema['properties']['selected_solutions']['maxItems'] == 2


def test_large_construction_preview_preserves_all_source_locators():
    events = []
    for index in range(2700):
        events.append({"id": f"h{index:05d}", "event": {"kind": "log", "session_id": f"session{index // 90}",
            "date": "2024-01-01 09:00", "content": "A source observation with detailed context. " * 30,
            "character_offset": 0}})
    rendered = mempal.construction_digest(events)
    value = json.loads(rendered)
    assert len(rendered) <= 160000
    assert value["total_events"] == 2700 and len(value["sessions"]) == 30
    assert {row[0] for session in value["sessions"] for row in session["observations"]} == {event["id"] for event in events}


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        return [1] * (sum(len(message["content"]) for message in messages) // 4 + 1)


class Client:
    def __init__(self, selected=None):
        self.calls = []
        self.selected = selected if selected is not None else ["S1", "S2"]

    def chat(self, messages, **options):
        self.calls.append((messages, options))
        return {"content": json.dumps({"analysis": "Source-supported choices.", "selected_solutions": self.selected}),
            "finish_reason": "stop", "usage": {}}


def test_runtime_causal_memory_neutral_equivalence_and_one_final_call():
    from ohpu.benchmarks import mempal_runtime as runtime
    from ohpu.experiments import pm3_official as official
    user = mempal.prepare_user("0000", native())
    baseline, neutral = Client(), Client()
    rows = runtime.answer_user(user, client=baseline, tokenizer=Tokenizer(), baseline="bm25_10")
    source = official.program("bm25", official.RECALL_SEARCH, top_k=10, max_chars=20000)
    other = runtime.answer_user(user, client=neutral, tokenizer=Tokenizer(), source=source)
    assert baseline.calls == neutral.calls
    assert len(rows) == len(other) == len(baseline.calls) == 2
    assert runtime.summarize_user(user, rows)["accuracy_pct_micro"] == 100
    first = json.dumps(baseline.calls[0][0])
    assert "FIRST_TARGET_ANSWER" not in first and "FUTURE_TARGET_ANSWER" not in first
    assert all(row["history_cutoff"] == case["history_cutoff"] for row, case in zip(rows, user["cases"]))
    state = runtime.MemPalState(user["user_id"], source)
    state.advance(user["events"][:user["cases"][0]["history_cutoff"]])
    state.context(None, "FIRST_TARGET_ANSWER")
    state.advance(user["events"][:user["cases"][1]["history_cutoff"]])
    evidence, _ = state.context(None, "FIRST_TARGET_ANSWER")
    assert "FIRST_TARGET_ANSWER" in evidence["output"] and "FUTURE_TARGET_ANSWER" not in evidence["output"]


def test_runtime_no_history_still_sees_current_public_logs():
    from ohpu.benchmarks import mempal_runtime as runtime
    user = mempal.prepare_user("0000", native())
    client = Client()
    rows = runtime.answer_user(user, client=client, tokenizer=Tokenizer(), baseline="no_history")
    assert all(row["history_count"] == 0 and not row["evidence"]["output"] for row in rows)
    assert "visible activity first" in json.dumps(client.calls[0][0])


def test_invalid_output_gets_score_floor_not_zero():
    from ohpu.benchmarks import mempal_runtime as runtime
    user = mempal.prepare_user("0000", native())
    rows = runtime.answer_user(user, client=Client(["S1", "S1"]), tokenizer=Tokenizer(), baseline="no_history")
    assert all(row["selection_score"] == -100 and row["format_error"] for row in rows)


@pytest.mark.parametrize("reason", ["length", "content_filter", None])
def test_incomplete_response_preserves_raw_evidence_and_actual_limits(reason):
    from ohpu.benchmarks import mempal_runtime as runtime
    user = mempal.prepare_user("0000", native())
    failures, rows = [], []
    class IncompleteClient(Client):
        def chat(self, messages, **options):
            result = super().chat(messages, **options)
            return {**result, "finish_reason": reason, "usage": {"completion_tokens": 512}}
    client = IncompleteClient()
    with pytest.raises(ValueError, match="Incomplete executor response"):
        runtime.answer_user(user, client=client, tokenizer=Tokenizer(), baseline="no_history",
            sink=rows.append, error_sink=failures.append)
    assert rows == [] and len(client.calls) == len(failures) == 1
    failure = failures[0]
    assert failure["code"] == "executor_incomplete" and failure["task_type"] == "solution_selection"
    assert failure["response"]["finish_reason"] == reason
    assert failure["response"]["usage"]["completion_tokens"] == 512
    assert json.loads(failure["response"]["content"])["selected_solutions"] == ["S1", "S2"]
    assert failure["limits"]["output_chars"] == 20000 and failure["limits"]["output_tokens"] == 512
    assert len(failure["messages_sha256"]) == 64


def test_transport_failure_does_not_attach_previous_response():
    from ohpu.benchmarks import mempal_runtime as runtime
    user = mempal.prepare_user("0000", native())
    failures, rows = [], []
    class FailingClient(Client):
        def chat(self, messages, **options):
            if self.calls:
                raise ConnectionError("Synthetic connection failure")
            return super().chat(messages, **options)
    with pytest.raises(ConnectionError):
        runtime.answer_user(user, client=FailingClient(), tokenizer=Tokenizer(), baseline="no_history",
            sink=rows.append, error_sink=failures.append)
    assert len(rows) == len(failures) == 1
    assert "response" not in failures[0]
    assert failures[0]["limits"]["output_chars"] == 20000


@pytest.mark.parametrize("analysis", ["line\nbreak", "line\rbreak", "tab\ttext", "\n" * 507,
    "\x0b" * 101, "\x00", "\x7f", "\x85", "\u2028", "\u2029"])
def test_analysis_schema_rejects_decoded_control_whitespace(analysis):
    import jsonschema
    from ohpu.benchmarks import mempal_runtime as runtime
    value = json.loads(json.dumps({"analysis": analysis, "selected_solutions": ["S1", "S2"]}))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, runtime.RESPONSE_FORMAT["json_schema"]["schema"])


def test_legacy_escaped_analysis_remains_scoreable_but_is_not_new_output():
    import jsonschema
    from ohpu.benchmarks import mempal_runtime as runtime
    value = {"analysis": 'A quote " and a backslash \\ are valid.', "selected_solutions": ["S2", "S8"]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, runtime.RESPONSE_FORMAT["json_schema"]["schema"])
    assert mempal.parse_output(json.dumps(value)) == ["S2", "S8"]


@pytest.mark.parametrize("replicates,selection_policy", [(1, "mean_gain"), (2, "mean_gain"), (1, "significant_vs_r0")])
def test_full_program_seed_and_frozen_train_test_pipeline(tmp_path, monkeypatch, replicates, selection_policy):
    import asyncio
    import importlib.util
    from pathlib import Path
    from ohpu.benchmarks import mempal_runtime as runtime
    path = Path(__file__).resolve().parents[2] / "analysis/skill_optimization_diagnosis/mempal_phg.py"
    specification = importlib.util.spec_from_file_location("mempal_pipeline_test", path)
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    monkeypatch.setattr(mempal, "require_local_storage", lambda path: Path(path).resolve())
    host = tmp_path / "host"
    host.mkdir()
    for name in driver.PHG.DRV.HOST_FILES:
        (host / name).write_text("")
    prepared = tmp_path / "prepared"
    users = [mempal.prepare_user(f"{index:04d}", native()) for index in range(100)]
    driver.write(prepared / "roles.json", mempal.roles([user["user_id"] for user in users]))
    driver.write(prepared / "summary.json", {"protocol": mempal.PROTOCOL, "users": {user["user_id"]: {} for user in users},
        "source": {"language": "en"}, "history_policy": "official sample order", "research_disclosure": "synthetic",
        "split": {"test": {"queries": 100}}})
    for user in users:
        driver.write(prepared / "users" / (user["user_id"] + ".json"), user)
    driver.write(prepared / "registration.json", {"files": {str(entry.relative_to(prepared)): driver.sha(entry)
        for entry in prepared.rglob("*") if entry.is_file()}})
    original_copy = driver.shutil.copytree
    def copytree(source, target, **options):
        if Path(source).name == "src":
            Path(target).mkdir(parents=True)
            return str(target)
        return original_copy(source, target, **options)
    monkeypatch.setattr(driver.shutil, "copytree", copytree)
    train = tmp_path / "train"
    plan = driver.register(train, prepared, host, iterations=1, replicates=replicates, selection_policy=selection_policy)
    assert plan["language"] == "en" and plan["program_edit_scope"] == "full-harness-v1"
    assert plan["phg"]["optimizer"]["reasoning_effort"] == "medium"
    assert plan["phg"]["replicates"] == replicates
    expected_protocol = driver.SINGLE_PASS_PROTOCOL if selection_policy == "significant_vs_r0" else driver.REPLICATED_PROTOCOL if replicates == 2 else driver.PROTOCOL
    assert plan["protocol"] == expected_protocol
    assert plan["arms"] == ["phg_r0", "phg_r1"]
    assert plan["upper_bounds"]["executor_requests"] == 200 * replicates
    assert len(plan["test_users"]) == len(plan["heldout_users"]) == 50
    assert plan["output_tokens"] == 512 and plan["host_contract"]["output_chars"] == 20000
    assert "FULL_PROGRAM = True" in (train / "generator/v0/generator.py").read_text()
    source = (train / "inputs/shared_control.py").read_text()
    actual, baseline = Client(), Client()
    runtime.answer_user(users[0], client=actual, tokenizer=Tokenizer(), source=source)
    runtime.answer_user(users[0], client=baseline, tokenizer=Tokenizer(), baseline="bm25_10")
    assert actual.calls == baseline.calls
    for user in plan["heldout_users"]:
        assert not (train / "private/native" / (user + ".json")).exists()
    with pytest.raises(ValueError, match="outside"):
        driver.score(train, "phg_r0", plan["heldout_users"][0])
    with pytest.raises(ValueError, match="outside"):
        driver.score(train, "bm25_10", plan["test_users"][0])
    with pytest.raises(FileNotFoundError):
        driver.register(tmp_path / "premature", prepared, train, reference=train)

    calls = []
    def executor(root):
        async def execute(action, arm, user):
            calls.append((root, action, arm, user))
            active = driver.read(root / "plan.json")
            assert user == "-" or user in active["test_users"]
            assert arm in active["phg"]["arms"]
            if action == "optimize":
                assert active["split"] == "TRAIN"
                original_copy(root / "generator/v0", root / "generator/v1")
                (root / "generator/v1/CHANGELOG.md").write_text("Synthetic candidate")
                driver.write(root / "generator/v1.json", {"version": 1, "parent": 0, "outcome": "accepted",
                    "spec_sha256": driver.PHG.spec_sha(root / "generator/v1")})
            elif action == "generate":
                driver.write(root / "harness" / arm / user / "harness.json", {"source": source,
                    "generation": {"valid": True, "usage": {}}})
            else:
                assert action == "score"
                native_user = driver.read(root / "private/native" / (user + ".json"))
                value = 50 if arm in ("phg_r1", "test_optimized") else -50
                rows = [{"user": user, "query_id": case["case_id"], "task_type": "solution_selection",
                    "accuracy": value, "selection_score": value, "request": mempal.request(case),
                    "prediction": ["S1", "S5"], "ground_truth": case["positive"], "format_error": None,
                    "evidence": {"output": "", "source_refs": []}, "guidance": ""} for case in native_user["cases"]]
                directory = root / "scores" / arm / user
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                driver.write(directory / "summary.json", {**runtime.summarize_user(native_user, rows),
                    "replicate_means": [value] * replicates})
        return execute

    asyncio.run(driver.run(train, execute=executor(train)))
    assert driver.read(train / "selection.json")["version"] == 1
    assert driver.read(train / "summary.json")["arms"]["phg_r0"]["mean"] == -50
    assert len(calls) == len(set(calls))
    if selection_policy == "significant_vs_r0":
        risk = driver.read(train / "diagnostics/round1/user_risk.json")
        assert risk["split"] == "TRAIN" and risk["complete"] and risk["wins"] == 50
        assert risk["positive_contribution"] == 100 and risk["negative_contribution"] == 0
        assert risk["top_five_positive_contribution"] == 10
        assert (train / "diagnostics/round1/r0_cases.json").exists()
        with pytest.raises(ValueError, match="requires exactly one"):
            driver.register(tmp_path / "bad-replicates", prepared, host, replicates=2, selection_policy=selection_policy)
    test = tmp_path / "test"
    heldout = driver.register(test, prepared, train, reference=train)
    assert heldout["test_users"] == plan["heldout_users"]
    assert heldout["arms"] == ["test_initial", "test_optimized"]
    assert heldout["upper_bounds"]["executor_requests"] == plan["scheduled_test"]["executor_requests"]
    assert heldout["phg"]["generator"] == plan["phg"]["generator"]
    assert heldout["host_contract"] == plan["host_contract"]
    assert heldout["phg"]["rounds"] == 0
    assert heldout["phg"]["replicates"] == replicates
    assert heldout["protocol"] == plan["protocol"]
    assert heldout["phg"]["selection_policy"] == selection_policy
    with pytest.raises(ValueError, match="inherit"):
        driver.register(tmp_path / "bad-test-policy", prepared, train, reference=train,
            selection_policy="mean_gain" if selection_policy == "significant_vs_r0" else "significant_vs_r0")
    asyncio.run(driver.run(test, execute=executor(test)))
    result = driver.read(test / "summary.json")
    assert set(result["arms"]) == {"test_initial", "test_optimized"}
    assert result["arms"]["test_optimized"]["mean"] == 50
    assert result["comparisons"]["test_initial"]["mean_pp"] == 100
    assert not (test / "optimizer").exists()
    with pytest.raises(ValueError, match="restricted to TRAIN"):
        driver.diagnose(test, 0)


@pytest.mark.parametrize("fail_second", [False, True])
def test_independent_generation_replicates_and_score_aggregation(tmp_path, monkeypatch, fail_second):
    from analysis.skill_optimization_diagnosis import mempal_phg as driver
    from ohpu.experiments import pm3_official as official
    user = mempal.prepare_user("0000", native())
    plan = {"split": "TRAIN", "seed": 20261026, "test_users": [user["user_id"]], "arms": ["phg_r0"],
        "phg": {"arms": {"phg_r0": 0}, "replicates": 2},
        "splits": {user["user_id"]: {"construction_cutoff": user["construction_cutoff"]}},
        "input_tokens": 32768, "output_tokens": 512}
    driver.write(tmp_path / "plan.json", plan)
    driver.write(tmp_path / "private/native" / (user["user_id"] + ".json"), user)
    driver.write(tmp_path / "private/users" / (user["user_id"] + ".json"), {"events": user["events"]})
    driver.initial_spec(tmp_path / "generator/v0")
    driver.write(tmp_path / "generator/v0.json", {"outcome": "accepted", "spec_sha256": driver.PHG.spec_sha(tmp_path / "generator/v0")})
    calls = []
    source = official.program("bm25", official.RECALL_SEARCH, top_k=10, max_chars=20000)
    def generated(root, arm, identifier, **options):
        calls.append((root, options["events"]))
        program = source + "\nREPLICATE_MARKER = " + repr(root.name) + "\n"
        valid = not (fail_second and root.name == "1")
        driver.write(root / "harness" / arm / identifier / "harness.json", {"source": program,
            "generation": {"valid": valid, "usage": {"llm_calls": 1}, "seconds": 1}})
        if not valid:
            raise RuntimeError("Synthetic failed second generation")
    monkeypatch.setattr(driver.PHG, "generate_harness", generated)
    if fail_second:
        with pytest.raises(RuntimeError, match="failed second generation"):
            driver.generate(tmp_path, "phg_r0", user["user_id"])
        receipt = driver.read(tmp_path / "harness/phg_r0" / user["user_id"] / "harness.json")
        assert receipt["generation"]["valid"] is False
        assert receipt["generation"]["usage"]["llm_calls"] == 2
        with pytest.raises(ValueError, match="no replacement"):
            driver.generate(tmp_path, "phg_r0", user["user_id"])
        assert len(calls) == 2
        return
    driver.generate(tmp_path, "phg_r0", user["user_id"])
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    assert calls[0][1] == calls[1][1] == user["events"][:user["construction_cutoff"]]
    receipt = driver.read(tmp_path / "harness/phg_r0" / user["user_id"] / "harness.json")
    assert receipt["generation"]["usage"]["llm_calls"] == 2
    assert receipt["replica_sources"][0] != receipt["replica_sources"][1]
    driver.generate(tmp_path, "phg_r0", user["user_id"])
    assert len(calls) == 2
    client = Client()
    driver.score(tmp_path, "phg_r0", user["user_id"], client=client, tokenizer=Tokenizer())
    summary = driver.read(tmp_path / "scores/phg_r0" / user["user_id"] / "summary.json")
    assert summary["replicate_means"] == [100, 100]
    assert summary["accuracy_pct_micro"] == 100
