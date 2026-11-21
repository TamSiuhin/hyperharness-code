import pytest

from analysis.skill_optimization_diagnosis.harness_permission import MARKER, validate_program


BACKBONE = """PROGRAM_API_VERSION = 1
BACKBONE = {'limit': 10}
#@USER_SECTION@#
def recall(context):
    return u_recall(context)
def apply(context):
    return user_guidance(context)
"""
SECTION = """USER_MODEL = {'interest': 'books'}
def u_recall(context):
    return {'query': 'books'}
def user_guidance(context):
    return USER_MODEL['interest']
"""


def test_section_accepts_executable_extensions():
    validate_program(BACKBONE.replace(MARKER, SECTION), BACKBONE, "section")


@pytest.mark.parametrize("replacement", [
    ("{'limit': 10}", "{'limit': 20}"),
    ("return u_recall(context)", "return {}"),
    ("return user_guidance(context)", "return {}"),
])
def test_section_rejects_shared_definition_changes(replacement):
    source = BACKBONE.replace(MARKER, SECTION).replace(*replacement)
    with pytest.raises(ValueError, match="shared definition"):
        validate_program(source, BACKBONE, "section")
    validate_program(source, BACKBONE, "full")


@pytest.mark.parametrize("addition", [
    "\ndef recall(context):\n    return {}\n",
    "\ndef other(context):\n    return {}\n",
    "\ndef u_modify(context):\n    alias = BACKBONE\n    alias['limit'] = 100\n    return {}\n",
    "\nu_shared = BACKBONE\n",
    "\nu_shared = recall.__globals__\n",
])
def test_section_rejects_escaping_its_boundary(addition):
    with pytest.raises(ValueError):
        validate_program(BACKBONE.replace(MARKER, SECTION) + addition, BACKBONE, "section")


def test_full_scope_can_choose_section_architecture():
    validate_program(BACKBONE.replace(MARKER, SECTION), BACKBONE, "full")


def test_section_requires_a_unique_marker():
    with pytest.raises(ValueError, match="exactly one"):
        validate_program(BACKBONE.replace(MARKER, SECTION), BACKBONE.replace(MARKER, ""), "section")


def test_host_checker_enforces_registered_scope(tmp_path):
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    spec = tmp_path / "spec"
    spec.mkdir()
    (spec / "backbone.py").write_text(BACKBONE)
    driver.write(tmp_path / "plan.json", {"permission_experiment": {
        "scope": "section", "validator_sha256": driver.sha(driver.HERE / "harness_permission.py")}})
    calls = []
    check = driver.permission_program_check(tmp_path, spec, lambda source: calls.append(source))
    good = BACKBONE.replace(MARKER, SECTION)
    bad = good.replace("return u_recall(context)", "return {}")
    assert check(good) is None
    assert check(bad).startswith("PermissionScopeError:")
    assert calls == [good]


def test_unregistered_runs_keep_original_checker(tmp_path):
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    driver.write(tmp_path / "plan.json", {})
    def check(source):
        return None
    assert driver.permission_program_check(tmp_path, tmp_path, check) is check
    driver.permission_spec_check(tmp_path, tmp_path)


def test_registered_generation_budget_cannot_expand(tmp_path):
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    spec = tmp_path / "spec"
    spec.mkdir()
    (spec / "backbone.py").write_text(BACKBONE)
    driver.write(tmp_path / "plan.json", {"phg": {"limits": {"llm_calls": 3},
        "open_generation_contract": True, "decision_model_enabled": False}, "permission_experiment": {
            "scope": "section", "validator_sha256": driver.sha(driver.HERE / "harness_permission.py"),
            "generation_caps": {"llm_calls": 3}}})
    driver.write(spec / "generation_policy.json", {"llm_calls": 3})
    driver.permission_spec_check(tmp_path, spec)
    (spec / "generation_policy.json").write_text('{"llm_calls": 4}')
    with pytest.raises(ValueError, match="cap exceeded"):
        driver.permission_spec_check(tmp_path, spec)


def test_pipeline_resume_skips_complete_failed_and_interrupted_users(tmp_path, monkeypatch):
    import asyncio
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    users = ["completed", "failed", "interrupted", "pending"]
    arm = "test_optimized"
    driver.write(tmp_path / "plan.json", {"test_users": users, "phg": {
        "arms": {arm: 0}, "luna_workers": 2}, "score_workers": 1})
    for user in users[:-1]:
        driver.write(tmp_path / "harness" / arm / user / "harness.json", {
            "generation": {"valid": user == "completed"}})
    driver.write(tmp_path / "scores" / arm / "completed/summary.json", {"original": True})
    preserved = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(driver.BASE.HISTORY.PILOT, "verify", lambda root: None)
    monkeypatch.setattr(driver, "freeze_selection", lambda root: None)
    async def no_training(*args, **options):
        return None
    monkeypatch.setattr(driver, "train_rounds", no_training)
    calls = []
    async def execute(action, method, user):
        calls.append((action, method, user))
    asyncio.run(driver.run_pipeline(tmp_path, execute=execute, score_stages=("score",),
        score_workers={"score": 1}, baseline_arms=(), control_arms=(),
        heldout_users=users, heldout_arms=(arm,), summary=lambda root: None))
    assert calls == [("generate", arm, "pending"), ("score", arm, "pending")]
    assert all(path.read_bytes() == content for path, content in preserved.items())


def _population_root(tmp_path, means, temperature=1.0):
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    users = ["a", "b"]
    total = 5
    driver.write(tmp_path / "plan.json", {"test_users": users, "external_baselines": {"arms": [], "files": {}}, "phg": {
        "arms": {f"phg_r{version}": version for version in range(total)}, "rounds": total,
        "population": {"seeds": ["direct", "bm25_evidence", "planned_bm25"], "temperature": temperature, "rng_seed": 7}}})
    for version, mean in means.items():
        spec = tmp_path / "generator" / f"v{version}"
        spec.mkdir(parents=True)
        (spec / "generator.py").write_text(f"VERSION = {version}\n")
        driver.write(tmp_path / "generator" / f"v{version}.json", {"version": version,
            "parent": None if version < 3 else 0, "outcome": "accepted", "spec_sha256": driver.spec_sha(spec)})
        for user in users[:1] if mean is None else users:
            driver.write(tmp_path / "scores" / f"phg_r{version}" / user / "summary.json", {
                "evaluation_complete": True, "runtime_violations": [], "accuracy_pct_micro": 0 if mean is None else mean})
    return driver, driver.read(tmp_path / "plan.json")


def test_population_softmax_sampling_is_reproducible_and_excludes_incomplete(tmp_path, monkeypatch):
    driver, plan = _population_root(tmp_path, {0: 30.0, 1: 28.0, 2: None})
    record = driver.sample_parent(tmp_path, plan, 3)
    expected = driver.softmax({0: 30.0, 1: 28.0}, 1.0)
    assert set(record["candidates"]) == {"0", "1"}
    assert abs(record["candidates"]["0"]["probability"] - expected[0]) < 1e-12
    assert record["parent"] in (0, 1) and record["lineage"] == ["direct", "bm25_evidence"][record["parent"]]
    assert driver.sample_parent(tmp_path, plan, 3) == record
    import random
    draw = random.Random("7:3").random()
    assert record["draw"] == draw and record["parent"] == (0 if draw < expected[0] else 1)


def test_population_temperature_controls_exploration():
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    greedy = driver.softmax({0: 30.0, 1: 25.0}, 1.0)
    flat = driver.softmax({0: 30.0, 1: 25.0}, 10.0)
    assert greedy[0] > 0.99 and 0.6 < flat[0] < 0.65
    assert abs(sum(flat.values()) - 1) < 1e-12


def test_population_summary_selects_best_complete_candidate(tmp_path, monkeypatch):
    driver, plan = _population_root(tmp_path, {0: 26.0, 1: 31.0, 2: None})
    monkeypatch.setattr(driver, "REFERENCES", ())
    for version in range(3):
        driver.population_round_summary(tmp_path, version)
    parent = driver.read(tmp_path / "rounds/parent.json")
    assert parent["version"] == 1
    assert [row["complete"] for row in parent["history"]] == [True, True, False]
    assert driver.read(tmp_path / "rounds/round2.json")["parent_decision"]["adopted"] is False


def test_population_requires_one_complete_seed(tmp_path, monkeypatch):
    driver, plan = _population_root(tmp_path, {0: None, 1: None, 2: None})
    monkeypatch.setattr(driver, "REFERENCES", ())
    driver.population_round_summary(tmp_path, 0)
    driver.population_round_summary(tmp_path, 1)
    with pytest.raises(RuntimeError, match="No seed has a complete"):
        driver.population_round_summary(tmp_path, 2)


def test_population_rounds_skip_seed_optimization_and_sample_before_each_child(tmp_path, monkeypatch):
    import asyncio
    driver, plan = _population_root(tmp_path, {0: 30.0, 1: 20.0, 2: 25.0})
    monkeypatch.setattr(driver, "REFERENCES", ())
    jobs = []
    async def job(action, arm, user):
        version = plan["phg"]["arms"][arm]
        assert (tmp_path / "rounds" / f"sample_v{version}.json").exists()
        jobs.append(arm)
        driver.write(tmp_path / "generator" / f"v{version}.json", {"version": version,
            "parent": driver.read(tmp_path / "rounds" / f"sample_v{version}.json")["parent"], "outcome": "unchanged"})
        return True
    evaluated = []
    async def evaluate(arm, user):
        evaluated.append(arm)
    asyncio.run(driver.train_rounds(tmp_path, plan, job=job, evaluate=evaluate, status=lambda value: None,
        diagnose_round=lambda root, version: None))
    assert jobs == ["phg_r3", "phg_r4"]
    assert sorted(set(evaluated)) == ["phg_r0", "phg_r1", "phg_r2"]
    assert driver.read(tmp_path / "rounds/parent.json")["version"] == 0


def test_population_experience_lists_every_earlier_candidate(tmp_path, monkeypatch):
    driver, plan = _population_root(tmp_path, {0: 30.0, 1: 28.0, 2: 27.0, 3: 31.0})
    monkeypatch.setattr(driver, "REFERENCES", ())
    for version in range(3):
        driver.population_round_summary(tmp_path, version)
    driver.sample_parent(tmp_path, plan, 3)
    driver.population_round_summary(tmp_path, 3)
    session = tmp_path / "optimizer/v3/workspace/controller_artifacts/controller"
    session.mkdir(parents=True)
    (session / "call-001.output.log").write_text("noise\ncodex\nworking\ntokens used\n1,234\nImproved retrieval queries.\n")
    target = tmp_path / "experience"
    driver.population_experience(tmp_path, plan, 4, target)
    record = driver.read(target / "population.json")
    assert [row["version"] for row in record["candidates"]] == [0, 1, 2, 3]
    assert [row["lineage"] for row in record["candidates"]] == ["direct", "bm25_evidence", "planned_bm25", "direct"]
    assert record["candidates"][3]["train_mean"] == 31.0
    assert record["candidates"][3]["optimizer_summary"] == "Improved retrieval queries."
    assert "VERSION = 3" in (target / "candidates/v3/diff_vs_parent.patch").read_text()
    assert (target / "candidates/v1/spec/generator.py").exists()
    assert [row["version"] for row in record["parent_samples"]] == [3]


def test_population_greedy_parent_picks_best_complete_candidate(tmp_path):
    driver, plan = _population_root(tmp_path, {0: 26.0, 1: 31.0, 2: None})
    plan["phg"]["population"]["parent_policy"] = "greedy"
    record = driver.sample_parent(tmp_path, plan, 3)
    assert record["parent"] == 1 and record["policy"] == "greedy" and set(record["candidates"]) == {"0", "1"}


def test_single_seed_incomplete_panel_starts_optimization_but_is_never_selected(tmp_path, monkeypatch):
    from analysis.skill_optimization_diagnosis import pm3_generator_harness as driver
    users = ["a", "b"]
    driver.write(tmp_path / "plan.json", {"test_users": users, "external_baselines": {"arms": [], "files": {}}, "phg": {
        "arms": {"phg_r0": 0, "phg_r1": 1}, "rounds": 2, "population": {"seeds": ["direct"], "parent_policy": "greedy",
            "temperature": 1.0, "rng_seed": 7, "incomplete_seed_parent": True}}})
    spec = tmp_path / "generator/v0"
    spec.mkdir(parents=True)
    (spec / "generator.py").write_text("VERSION = 0\n")
    driver.write(tmp_path / "generator/v0.json", {"version": 0, "parent": None, "outcome": "accepted", "spec_sha256": driver.spec_sha(spec)})
    driver.write(tmp_path / "scores/phg_r0/a/summary.json", {"evaluation_complete": True, "runtime_violations": [], "accuracy_pct_micro": 40})
    monkeypatch.setattr(driver, "REFERENCES", ())
    plan = driver.read(tmp_path / "plan.json")
    driver.population_round_summary(tmp_path, 0)
    assert driver.read(tmp_path / "rounds/parent.json")["version"] is None
    record = driver.sample_parent(tmp_path, plan, 1)
    assert record["parent"] == 0 and record["parent_panel_complete"] is False
    with pytest.raises(ValueError, match="No valid TRAIN parent"):
        driver.freeze_selection(tmp_path)


@pytest.mark.parametrize("recorded", [True, False])
def test_population_recorded_optimizer_failure_skips_round(tmp_path, monkeypatch, recorded):
    import asyncio
    driver, plan = _population_root(tmp_path, {0: 30.0, 1: 20.0, 2: 25.0})
    plan["phg"]["population"]["parent_policy"] = "greedy"
    monkeypatch.setattr(driver, "REFERENCES", ())
    jobs = []
    async def job(action, arm, user):
        version = plan["phg"]["arms"][arm]
        jobs.append(arm)
        if version == 3 and recorded:
            driver.write(tmp_path / "generator/v3.json", {"version": 3, "parent": 0, "outcome": "failed", "error": "timeout"})
        if version == 3:
            return False
        driver.write(tmp_path / "generator" / f"v{version}.json", {"version": version, "parent": 0, "outcome": "unchanged"})
        return True
    async def evaluate(arm, user):
        pass
    run = driver.train_rounds(tmp_path, plan, job=job, evaluate=evaluate, status=lambda value: None,
        diagnose_round=lambda root, version: None)
    if recorded:
        asyncio.run(run)
        assert jobs == ["phg_r3", "phg_r4"]
    else:
        with pytest.raises(RuntimeError, match="Optimizer failed"):
            asyncio.run(run)
