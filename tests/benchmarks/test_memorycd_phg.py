import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from ohpu.benchmarks import memorycd as data, perma


def adapter():
    path = Path(__file__).resolve().parents[2] / "analysis/skill_optimization_diagnosis/memorycd_phg.py"
    spec = importlib.util.spec_from_file_location("memorycd_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_selection_and_test_handoff_are_user_disjoint(tmp_path, monkeypatch):
    driver = adapter()
    monkeypatch.setattr(perma, "require_local_storage", lambda path: Path(path).resolve())
    source = tmp_path / "source"
    source.mkdir()
    for name in driver.PHG.DRV.HOST_FILES:
        (source / name).write_text("")
    records = [{"user_id": f"user{index}", "interactions": {"Books": [
        {"parent_asin": "old", "timestamp": 1, "rating": 4, "text": "I like practical\u2028products"}],
        "Home_and_Kitchen": [{"parent_asin": "new", "timestamp": 2, "rating": 5,
            "title": "Useful product", "text": "This product is useful", "negative_parent_asins": ["other"]}]}} for index in range(4)]
    metadata = {name: {"title": name} for name in ("old", "new", "other")}
    monkeypatch.setattr(data, "load_source", lambda path: (records, metadata, {}))
    prepared = tmp_path / "prepared"
    data.prepare_dataset(source, prepared, train_users=2, num_test=1)
    original_copy = driver.shutil.copytree

    def copytree(source, target, **kwargs):
        if Path(source).name == "src":
            Path(target).mkdir(parents=True)
            return str(target)
        return original_copy(source, target, **kwargs)

    monkeypatch.setattr(driver.shutil, "copytree", copytree)
    out = tmp_path / "train"
    plan = driver.register(out, prepared, source, iterations=1)
    assert plan["protocol"].endswith("v8")
    assert plan["phg"]["generator"]["model"] == "gpt-6-sol"
    assert plan["task_guard"]["reference_version"] == 0
    assert plan["output_tokens"] == 4096
    assert plan["text_output_limits"] == {"review_summarization": 160, "review_generation": 3000}
    assert plan["decoding_contract"]["output_repair"] is False
    assert plan["decoding_contract"]["nonempty_strings"] is True
    assert plan["decoding_contract"]["mode"] == "guidance-json-schema-v1"
    assert plan["tasks"] == list(data.TASKS)
    assert set(plan["task_weights"].values()) == {0.25}
    assert plan["arms"] == ["phg_r0", "phg_r1"]
    assert plan["upper_bounds"]["requests_per_arm"] == 8
    assert plan["upper_bounds"]["executor_requests"] == 16
    prompt = (out / "generator/v0/prompt.md").read_text()
    assert "no defaults, annotations" in prompt
    assert "Never pass function names as values" in prompt
    assert plan["phg"]["generator"]["request_timeout_seconds"] == 600
    assert plan["phg"]["limits"]["wall_seconds"] == 7800
    assert plan["phg"]["generator"]["transport_retries"] == 1
    for user in plan["heldout_users"]:
        assert not (out / "private/native" / f"{user}.json").exists()
    with pytest.raises(ValueError, match="outside"):
        driver.score(out, "phg_r0", plan["heldout_users"][0])
    with pytest.raises(ValueError, match="outside"):
        driver.score(out, "no_history", plan["test_users"][0])

    def executor(root):
        async def execute(action, arm, user):
            active = driver.read(root / "plan.json")
            assert user == "-" or user in active["test_users"]
            assert arm in active["phg"]["arms"]
            if action == "optimize":
                assert active["split"] == "TRAIN"
                original_copy(root / "generator/v0", root / "generator/v1")
                (root / "generator/v1/CHANGELOG.md").write_text("Test-only candidate")
                driver.write(root / "generator/v1.json", {"version": 1, "parent": 0, "outcome": "accepted",
                    "spec_sha256": driver.PHG.spec_sha(root / "generator/v1")})
            elif action == "generate":
                driver.write(root / "harness" / arm / user / "harness.json", {
                    "source": (root / "generator/v0/control_program.py").read_text(),
                    "generation": {"valid": True, "usage": {}}})
            else:
                assert action == "score"
                value = 60 if arm in ("phg_r1", "test_optimized") else 20
                directory = root / "scores" / arm / user
                driver.write(directory / "summary.json", {"evaluation_complete": True, "runtime_violations": [],
                    "accuracy_pct_micro": value, "rating_metrics": {"mae": 1, "rmse": 1}, "invalid_outputs": 0,
                    "task_scores": {task: value for task in data.TASKS},
                    "ranking_metrics": data.ranking_metrics(["new"], "new"),
                    "text_metrics": {task: data.text_metrics("Useful product", "Useful product") for task in data.TEXT_OUTPUTS}})
                (directory / "records.jsonl").write_text("".join(json.dumps({"user": user,
                    "query_id": query, "task_type": query.split(":")[-1], "accuracy": value,
                    "request": "test request", "agent_response": "test\u2028answer"}, ensure_ascii=False) + "\n"
                    for query in active["splits"][user]["evaluated"]))
        return execute

    asyncio.run(driver.run(out, execute=executor(out)))
    assert driver.read(out / "selection.json")["version"] == 1
    assert driver.read(out / "operations/status.json")["phase"] == "completed"
    assert {row["task_type"] for row in driver.read(out / "diagnostics/round1/cases.json")} == set(data.TASKS)
    test = tmp_path / "test"
    test_plan = driver.register(test, prepared, out, reference=out)
    assert test_plan["test_users"] == plan["heldout_users"]
    assert set(test_plan["test_users"]).isdisjoint(plan["test_users"])
    assert test_plan["arms"] == ["test_initial", "test_optimized"]
    assert test_plan["phg"]["rounds"] == 0
    assert test_plan["phg"]["generator"] == plan["phg"]["generator"]
    assert test_plan["tasks"] == plan["tasks"]
    assert test_plan["task_weights"] == plan["task_weights"]
    assert test_plan["task_guard"] == plan["task_guard"]
    assert test_plan["text_output_limits"] == plan["text_output_limits"]
    assert test_plan["output_tokens"] == plan["output_tokens"]
    assert test_plan["decoding_contract"] == plan["decoding_contract"]
    assert test_plan["upper_bounds"]["sol_sessions"] == 0
    assert plan["scheduled_test"]["executor_requests"] == test_plan["upper_bounds"]["executor_requests"]
    assert driver.sha(test / "memorycd_phg.py") == driver.sha(out / "memorycd_phg.py")
    asyncio.run(driver.run(test, execute=executor(test)))
    summary = driver.read(test / "summary.json")
    assert set(summary["arms"]) == {"test_initial", "test_optimized"}
    assert summary["arms"]["test_optimized"]["mean"] == 60
    assert set(summary["comparisons"]) == {"test_initial"}
    assert set(summary["task_comparisons"]["test_initial"]) == set(data.TASKS)
    assert summary["arms"]["test_optimized"]["task_means"] == dict.fromkeys(data.TASKS, 60)
    assert driver.read(test / "operations/status.json")["phase"] == "completed"
    with pytest.raises(ValueError, match="restricted to TRAIN"):
        driver.diagnose(test, 0)


def test_initial_digest_is_bounded_but_preserves_full_history_access(tmp_path):
    driver = adapter()
    driver.initial_spec(tmp_path / "spec")
    specification = importlib.util.spec_from_file_location("memorycd_digest_test", tmp_path / "spec/history_adapter.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    events = [{"id": f"h{index}", "event": {"review": "evidence " * 1000,
        "rating": 1 if index < 25 else 5, "review_title": "Brief title", "domain": "Books"}} for index in range(50)]
    result = json.loads(module.digest(events))
    assert result["total_history_events"] == 50
    assert 0 < result["selected_events"] < 50
    assert len(json.dumps(result)) < 81000
    assert result["events"][-1]["id"] == "h49"
    assert "fullhistory remains available" in result["selection"]
    assert len(events) == 50
    assert result["history_statistics"]["rating_count"] == 50
    assert result["history_statistics"]["rating_mean"] == 3
    assert result["history_statistics"]["rating_counts"] == {"1.0": 25, "5.0": 25}
    assert result["history_statistics"]["title_word_bins"]["1-2"] == 50
    assert result["history_statistics"]["domain_counts"] == {"Books": 50}
    generator_spec = importlib.util.spec_from_file_location("memorycd_initial_generator", tmp_path / "spec/generator.py")
    generator = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator)
    assert generator.REPAIRS == 4


def test_initial_backbone_routes_all_hooks_by_actual_task(tmp_path):
    driver = adapter()
    driver.initial_spec(tmp_path / "spec")
    namespace = {}
    exec((tmp_path / "spec/control_program.py").read_text(), namespace)
    seen = []
    def route(request, kind, prior):
        seen.append(("route", kind))
        return {"history": True, "queries": [request]}
    def keep(text, app, kind):
        seen.append(("keep", kind))
        return True
    def guidance(request, kind, history):
        seen.append(("guidance", kind))
        return kind
    namespace.update(user_route=route, user_keep=keep, user_guidance=guidance)
    for task in data.TASKS:
        initial = namespace["recall"]({"stage": "context", "task_kind": task, "query": "product",
            "backend": {"step": 0}})["memory_query"]
        namespace["recall"]({"stage": "context", "task_kind": task, "query": "product", "backend": {
            "step": 1, "state": initial["state"], "result": {"data": [{"ref": "record:observations:test", "text": "evidence"}]}}})
        result = namespace["apply"]({"stage": "generation", "task_kind": task, "query": "product",
            "base": {"expressible": ["prefix"]}, "backend_evidence": {"source_refs": ["record:observations:test"]}})
        assert result == {"prefix": task}
        assert seen[-3:] == [("route", task), ("keep", task), ("guidance", task)]


def test_task_guard_rejects_average_gain_that_hurts_one_task(tmp_path):
    driver = adapter()
    driver.write(tmp_path / "plan.json", {"dataset": "memorycd", "split": "TRAIN", "test_users": ["user"], "tasks": list(data.TASKS),
        "external_baselines": {"arms": [], "files": {}},
        "task_guard": {"numerical_tolerance": 1e-9}, "phg": {"arms": {f"phg_r{version}": version for version in range(4)}}})
    for version, values in enumerate(([50, 50, 50, 50], [70, 60, 49, 60], [51, 51, 50, 51])):
        driver.write(tmp_path / "generator" / f"v{version}.json", {"version": version})
        driver.write(tmp_path / "scores" / f"phg_r{version}" / "user/summary.json", {
            "evaluation_complete": True, "runtime_violations": [], "accuracy_pct_micro": sum(values) / 4,
            "task_scores": dict(zip(data.TASKS, values))})
        summary = driver.round_summary(tmp_path, version)
        assert summary["parent_decision"]["adopted"] == (version != 1)
        if version == 1:
            assert summary["parent_decision"]["mean_versus_parent"] > 0
            assert summary["parent_decision"]["regressed_tasks"] == ["review_summarization"]
    assert driver.read(tmp_path / "rounds/parent.json")["version"] == 2
    assert not summary["parent_decision"]["all_tasks_strictly_improved"]
    driver.write(tmp_path / "generator/v3.json", {"version": 3})
    incomplete = driver.round_summary(tmp_path, 3)
    assert not incomplete["parent_decision"]["adopted"]
    assert driver.read(tmp_path / "rounds/parent.json")["version"] == 2
    plan = driver.read(tmp_path / "plan.json")
    test_out = tmp_path / "heldout"
    driver.write(test_out / "plan.json", {**plan, "split": "TEST"})
    with pytest.raises(ValueError, match="restricted to TRAIN"):
        driver.round_summary(test_out, 3)


def test_study_cli_updates_status_and_completes_test_handoff(tmp_path, monkeypatch):
    driver = adapter()
    train, test = tmp_path / "train", tmp_path / "test"
    driver.write(train / "plan.json", {"split": "TRAIN", "prepared": str(tmp_path / "prepared")})
    driver.write(train / "summary.json", {"split": "TRAIN"})
    driver.write(train / "operations/study_status.json", {"phase": "previous_status"})
    monkeypatch.setattr(driver, "host", lambda root: None)
    monkeypatch.setattr(driver.PHG.BASE.HISTORY.PILOT, "verify", lambda root: None)
    monkeypatch.setattr(driver.sys, "argv", ["memorycd_phg.py", "study", "--out", str(train), "--test-out", str(test)])

    async def run(root):
        assert root == train

    def register(root, prepared, host, **kwargs):
        assert root == test and kwargs["reference"] == train
        driver.write(test / "registration.json", {"files": {}})

    def execute(command, **kwargs):
        assert driver.read(train / "operations/study_status.json")["phase"] == "test_evaluation"
        driver.write(test / "summary.json", {"split": "TEST"})
        driver.write(test / "operations/status.json", {"phase": "completed_with_failures"})

    monkeypatch.setattr(driver, "run", run)
    monkeypatch.setattr(driver, "register", register)
    monkeypatch.setattr(driver.subprocess, "run", execute)
    driver.main()
    assert driver.read(train / "operations/study_status.json")["phase"] == "completed_with_failures"
    assert driver.read(train / "study_summary.json")["test"]["split"] == "TEST"


def test_monitor_records_failed_slots_and_dead_controller_without_retry(tmp_path, monkeypatch):
    driver = adapter()
    train, test = tmp_path / "train", tmp_path / "test"
    driver.write(train / "plan.json", {"test_users": ["user"], "arms": ["phg_r0"]})
    driver.write(train / "operations/status.json", {"phase": "attention_required", "failures": ["generate/phg_r0/user"]})
    driver.write(train / "operations/controller.json", {"pid": 123456})

    def missing(pid, signal):
        assert signal == 0
        raise ProcessLookupError()

    monkeypatch.setattr(driver.os, "kill", missing)
    first = driver.record_health(train, test)
    assert first["runs"]["TEST"] == {"registered": False}
    assert {row["kind"] for row in first["alerts"]} == {"failed_slots", "unexpected_controller_exit", "attention_required"}
    driver.record_health(train, test)
    assert len(driver.records(train / "operations/alerts.jsonl")) == 3


def test_monitor_reports_stalled_live_controller_and_tolerates_partial_json(tmp_path, monkeypatch):
    driver = adapter()
    driver.write(tmp_path / "plan.json", {"test_users": ["user"], "arms": ["phg_r0"],
        "monitoring": {"stalled_seconds": 5}})
    driver.write(tmp_path / "operations/status.json", {"phase": "train_round_0", "failures": []})
    driver.write(tmp_path / "operations/controller.json", {"pid": 123456})
    path = tmp_path / "scores/phg_r0/user/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text("{")
    monkeypatch.setattr(driver.os, "kill", lambda pid, signal: None)
    report = driver.health_snapshot(tmp_path, tmp_path / "future_test", timestamp=path.stat().st_mtime + 10)
    assert report["runs"]["TRAIN"]["unreadable_files"] == ["scores/phg_r0/user/summary.json"]
    assert any(row["kind"] == "no_progress" for row in report["alerts"])
