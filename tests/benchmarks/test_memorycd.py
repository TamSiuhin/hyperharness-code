import copy
import json

import pytest

from ohpu.benchmarks import memorycd as data


def fixture_user():
    metadata = {name: {"title": f"product {name}", "average_rating": "PRIVATE_AGGREGATE"}
        for name in ("past", "future", "target", "negative")}
    record = {"user_id": "user", "interactions": {"Books": [
        {"parent_asin": "past", "timestamp": 1, "rating": 4, "text": "past evidence"},
        {"parent_asin": "future", "timestamp": 10, "rating": 1, "text": "PRIVATE_FUTURE"}],
        "Home_and_Kitchen": [{"parent_asin": "target", "timestamp": 10, "rating": 5,
            "title": "PRIVATE_REVIEW_TITLE", "text": "PRIVATE_REVIEW_BODY", "negative_parent_asins": ["negative"]}]}}
    return record, metadata


def test_split_exact_100_and_all_remaining_users():
    users = [f"user{index:03}" for index in range(323)]
    split = data.roles(users)
    assert len(split["train"]) == 100 and len(split["test"]) == 223
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["train"]) | set(split["test"]) == set(users)
    assert data.roles(list(reversed(users))) == split


def test_causal_history_and_target_blind_requests():
    record, metadata = fixture_user()
    original = copy.deepcopy(record)
    user = data.prepare_user(record, metadata, num_test=1)
    assert record == original
    assert len(user["history"]) == 1
    assert user["excluded_noncausal_source_interactions"] == 1
    case = user["cases"][0]
    assert case["gold_review"] == "PRIVATE_REVIEW_BODY"
    assert case["gold_review_title"] == "PRIVATE_REVIEW_TITLE"
    assert "PRIVATE_" not in json.dumps(data.history_events(user))
    for task in ("item_ranking", "rating_prediction"):
        visible = data.request(case, task)
        assert "PRIVATE_" not in visible
        assert "gold" not in visible
    changed = {**case, "gold_item_id": "negative", "gold_rating": 1}
    assert data.request(case, "item_ranking") == data.request(changed, "item_ranking")


@pytest.mark.parametrize("text,task", [
    ('{"rating":true}', "rating_prediction"),
    ('{"rating":3,"rating":4}', "rating_prediction"),
    ('{"rating":3.5}', "rating_prediction"),
    ('{"ranking":["target","target"]}', "item_ranking"),
    ('{"ranking":["target"]}', "item_ranking"),
])
def test_invalid_outputs_are_never_repaired(text, task):
    record, metadata = fixture_user()
    case = data.prepare_user(record, metadata, num_test=1)["cases"][0]
    with pytest.raises(ValueError):
        data.parse_output(text, task, case)


def test_native_rank_and_rating_metrics():
    record, metadata = fixture_user()
    case = data.prepare_user(record, metadata, num_test=1)["cases"][0]
    assert data.parse_output('{"rating":5}', "rating_prediction", case) == 5
    ranking = data.parse_output('{"ranking":["negative","target"]}', "item_ranking", case)
    assert data.ranking_metrics(ranking, "target")["recall@1"] == 0
    assert data.ranking_metrics(ranking, "target")["ndcg@5"] == pytest.approx(0.6309297536)


def test_preparation_is_immutable_and_preserves_every_user(tmp_path, monkeypatch):
    record, metadata = fixture_user()
    records = [{**copy.deepcopy(record), "user_id": f"user{index}"} for index in range(4)]
    monkeypatch.setattr(data, "load_source", lambda source: (records, metadata, {"revision": data.DATASET_REVISION}))
    out = tmp_path / "prepared"
    summary = data.prepare_dataset(tmp_path, out, train_users=1, num_test=1)
    assert summary["queries"]["train"]["model_requests"] == 4
    assert summary["tasks"] == list(data.TASKS)
    assert summary["queries"]["test"]["users"] == 3
    registration = json.loads((out / "registration.json").read_text())
    assert registration["files"]["users.jsonl"] == data.file_sha(out / "users.jsonl")
    with pytest.raises(ValueError, match="fresh"):
        data.prepare_dataset(tmp_path, out, train_users=1, num_test=1)


def test_missing_negative_metadata_is_audited_without_replacing_user_or_target():
    record, metadata = fixture_user()
    record["interactions"][data.TARGET_DOMAIN][0]["negative_parent_asins"].append("unknown")
    catalog = {item: {"title": item, "category": "category"} for item in ("target", "negative")}
    user = data.prepare_user(record, metadata, num_test=1, candidate_metadata=catalog)
    case = user["cases"][0]
    assert case["gold_item_id"] == "target"
    assert case["excluded_untitled_negatives"] == ["unknown"]
    assert len(case["candidates"]) == 2
    assert all(set(item) == {"item_id", "title", "category"} for item in case["candidates"])


def test_actual_runtime_keeps_queries_label_free_and_calls_executor_once():
    from ohpu.benchmarks.memorycd_runtime import answer_user, summarize_user, validate_harness
    from ohpu.experiments import pm3_official

    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    prompts = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": list(range(sum(len(message["content"]) for message in messages)))}

    class Client:
        def chat(self, messages, **kwargs):
            prompts.append(messages)
            task = kwargs["response_format"]["json_schema"]["name"].removeprefix("memorycd_")
            outputs = {"item_ranking": {"ranking": ["target", "negative"]}, "rating_prediction": {"rating": 4},
                "review_summarization": {"title": "PRIVATE_REVIEW_TITLE"},
                "review_generation": {"review": "PRIVATE_REVIEW_BODY"}}
            return {"content": json.dumps(outputs[task]),
                "finish_reason": "stop", "usage": {}}

    rows = answer_user(user, baseline="no_history", client=Client(), tokenizer=Tokenizer())
    summary = summarize_user(user, rows)
    assert len(prompts) == 4
    assert "PRIVATE_" not in json.dumps(prompts[:2] + prompts[3:])
    assert "PRIVATE_REVIEW_TITLE" not in json.dumps(prompts)
    assert summary["accuracy_pct_micro"] == 93.75
    assert summary["task_scores"] == {"item_ranking": 100, "rating_prediction": 75,
        "review_summarization": 100, "review_generation": 100}
    assert summary["rating_metrics"]["mae"] == 1
    assert summary["text_metrics"]["review_generation"]["rougeL"] == 1
    source = pm3_official.uniform_programs()["bm25_10"]
    validate_harness(source, data.history_events(user))
    changed = copy.deepcopy(user)
    changed["cases"][0]["gold_item_id"] = "negative"
    changed["cases"][0]["gold_rating"] = 1
    changed["cases"][0]["gold_review_title"] = "different title"
    first = answer_user(user, source=source, client=Client(), tokenizer=Tokenizer())
    second = answer_user(changed, source=source, client=Client(), tokenizer=Tokenizer())
    assert first[0]["evidence"] == second[0]["evidence"]
    assert first[0]["request"] == second[0]["request"]
    assert prompts[-8:-6] == prompts[-4:-2]
    assert "PRIVATE_REVIEW_BODY" not in json.dumps(prompts[-1])
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_user(user, rows[:2])


@pytest.mark.parametrize("task", data.TASKS)
def test_every_task_contributes_equally_to_selection_score(task):
    from ohpu.benchmarks.memorycd_runtime import summarize_user

    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    rows = [{"query_id": "user:0:" + name, "task_type": name, "accuracy": 100, "format_error": None,
        "metrics": data.ranking_metrics(["target"], "target") if name == "item_ranking" else
            {"absolute_error": 0, "squared_error": 0} if name == "rating_prediction" else
            {"rouge1": 1, "rougeL": 1, "bleu1": 1, "bleu4": 1}} for name in data.TASKS]
    assert summarize_user(user, rows)["accuracy_pct_micro"] == 100
    next(row for row in rows if row["task_type"] == task)["accuracy"] = 0
    assert summarize_user(user, rows)["accuracy_pct_micro"] == 75


def test_decoding_schema_does_not_depend_on_ground_truth():
    record, metadata = fixture_user()
    case = data.prepare_user(record, metadata, num_test=1)["cases"][0]
    altered = {**case, "gold_item_id": "negative", "gold_rating": 1}
    schema = data.output_schema(case, "item_ranking")
    assert schema == data.output_schema(altered, "item_ranking")
    field = schema["json_schema"]["schema"]["properties"]["ranking"]
    assert field["minItems"] == field["maxItems"] == 2
    assert set(field["items"]["enum"]) == {"target", "negative"}


def test_top_five_has_native_metrics_without_filling_lower_ranks():
    case = {"candidates": [{"item_id": str(index)} for index in range(20)]}
    prediction = data.parse_output('{"ranking":["3","7","1","9","2"]}', "item_ranking", case)
    assert data.ranking_metrics(prediction, "15")["ndcg@5"] == 0
    assert data.ranking_metrics(prediction, "7")["ndcg@5"] == pytest.approx(0.6309297536)
    assert data.output_schema(case, "item_ranking")["json_schema"]["schema"]["properties"]["ranking"]["maxItems"] == 5


def test_jsonl_reader_preserves_unicode_line_separators_in_reviews(tmp_path):
    rows = [{"text": "first\u2028second\u0085third"}, {"text": "another review"}]
    path = tmp_path / "reviews.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    assert data.read_records(path) == rows


def test_text_tasks_only_expose_official_task_inputs():
    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    case = user["cases"][0]
    summary = data.request(case, "review_summarization")
    review = data.request(case, "review_generation")
    assert "PRIVATE_REVIEW_BODY" in summary and "PRIVATE_REVIEW_TITLE" not in summary
    assert "PRIVATE_" not in review
    assert json.loads(summary.split("\n", 1)[1])["rating"] == 5
    assert json.loads(review.split("\n", 1)[1])["rating"] == 5
    changed = {**case, "gold_review_title": "another title", "gold_item_id": "negative"}
    assert summary == data.request(changed, "review_summarization")
    changed["gold_review"] = "another body"
    assert review == data.request(changed, "review_generation")
    for task in data.TASKS:
        assert data.output_schema(case, task) == data.output_schema(changed, task)


@pytest.mark.parametrize("task,key", [("review_summarization", "title"), ("review_generation", "review")])
def test_text_output_schema_and_validation(task, key):
    assert data.parse_output(json.dumps({key: "Actual text"}), task, {}) == "Actual text"
    assert data.output_schema({}, task)["json_schema"]["schema"]["required"] == [key]
    for value in ({key: " "}, {key: 4}, {key: "text", "extra": True}):
        with pytest.raises(ValueError):
            data.parse_output(json.dumps(value), task, {})
    field = data.output_schema({}, task)["json_schema"]["schema"]["properties"][key]
    assert field["maxLength"] == data.TEXT_OUTPUT_LIMITS[task]
    assert "pattern" not in field
    with pytest.raises(ValueError):
        data.parse_output(json.dumps({key: "x" * (data.TEXT_OUTPUT_LIMITS[task] + 1)}), task, {})


def test_incomplete_executor_response_is_retained_without_retry():
    from ohpu.benchmarks.memorycd_runtime import answer_user
    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    failures, calls = [], []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]
    class Client:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            return {"finish_reason": "length", "content": "unfinished response", "usage": {"completion_tokens": 4096}}
    with pytest.raises(ValueError, match="finish_reason=length"):
        answer_user(user, baseline="no_history", client=Client(), tokenizer=Tokenizer(), error_sink=failures.append)
    assert len(calls) == len(failures) == 1
    assert failures[0]["response"] == "unfinished response"
    assert failures[0]["usage"]["completion_tokens"] == failures[0]["output_token_limit"] == 4096


@pytest.mark.parametrize("kind", ["input", "output"])
def test_opt_in_filter_skip_continues_and_preserves_task_weights(kind):
    from ohpu.benchmarks import memorycd_runtime as runtime
    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    user["cases"].append({**user["cases"][0], "case_id": "user:1"})
    user["filter_policy"] = runtime.FILTER_PROTOCOL
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]
    class Client:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                if kind == "input":
                    error = RuntimeError("provider rejected input")
                    error.body = {"error": {"innererror": {"code": "ResponsibleAIPolicyViolation"}}}
                    raise error
                return {"finish_reason": "content_filter", "usage": {}}
            task = kwargs["response_format"]["json_schema"]["name"].removeprefix("memorycd_")
            outputs = {"item_ranking": {"ranking": ["target", "negative"]},
                "rating_prediction": {"rating": 5}, "review_summarization": {"title": "PRIVATE_REVIEW_TITLE"},
                "review_generation": {"review": "PRIVATE_REVIEW_BODY"}}
            return {"finish_reason": "stop", "content": json.dumps(outputs[task]), "usage": {}}
    client = Client()
    rows = runtime.answer_user(user, baseline="no_history", client=client, tokenizer=Tokenizer())
    assert client.calls == len(rows) == 8
    summary = runtime.summarize_user(user, rows)
    assert summary["accuracy_pct_micro"] == 100
    assert summary["filtered_queries"] == 1 and summary["coverage"] == 7 / 8
    assert not summary["original_evaluation_complete"]
    assert runtime.answer_user(user, baseline="no_history", client=client, tokenizer=Tokenizer(), retained=rows) == rows
    assert client.calls == 8
    user["known_filtered_queries"] = [rows[0]["query_id"]]
    resumed = runtime.answer_user(user, baseline="no_history", client=client, tokenizer=Tokenizer())
    assert client.calls == 15 and resumed[0]["executor_calls"] == 0
    assert runtime.summarize_user(user, resumed)["filtered_queries"] == 1
    strict = {key: value for key, value in user.items() if key != "filter_policy"}
    with pytest.raises(ValueError, match="registered skip protocol"):
        runtime.summarize_user(strict, rows)
    rows[1] = {**rows[1], "filtered": True}
    with pytest.raises(ValueError, match="Each MemoryCD task"):
        runtime.summarize_user(user, rows)


@pytest.mark.parametrize("body", [None, {"code": "429"}, {"code": "service_unavailable_error"}])
def test_skip_policy_never_masks_nonfilter_errors(body):
    from ohpu.benchmarks import memorycd_runtime as runtime
    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    user["filter_policy"] = runtime.FILTER_PROTOCOL
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]
    class Client:
        def chat(self, messages, **kwargs):
            error = RuntimeError("content_filter mentioned but not a structured provider rejection")
            error.body = body
            raise error
    with pytest.raises(RuntimeError):
        runtime.answer_user(user, baseline="no_history", client=Client(), tokenizer=Tokenizer())


def test_common_samples_prevent_missingness_selection_gain():
    from ohpu.benchmarks.memorycd_runtime import common_sample_scores
    left = [{"query_id": f"{task}:{index}", "task_type": task, "accuracy": score}
        for task in data.TASKS for index, score in enumerate([0, 100])]
    right = copy.deepcopy(left)
    left[0] = {**left[0], "filtered": True, "accuracy": None}
    result = common_sample_scores(left, right)
    assert result["left"] == result["right"]
    assert result["queries"] == 7
    right[1] = {**right[1], "filtered": True, "accuracy": None}
    with pytest.raises(ValueError, match="Each MemoryCD task"):
        common_sample_scores(left, right)


@pytest.mark.parametrize("missing_task", data.TASKS)
def test_sparse_filter_policy_retains_user_and_marks_empty_task(missing_task):
    from ohpu.benchmarks import memorycd_runtime as runtime
    record, metadata = fixture_user()
    user = data.prepare_user(record, metadata, num_test=1)
    rows = [{"query_id": "user:0:" + task, "task_type": task, "accuracy": 100, "format_error": None,
        "metrics": data.ranking_metrics(["target"], "target") if task == "item_ranking" else
        {"absolute_error": 0, "squared_error": 0} if task == "rating_prediction" else
        {"rouge1": 1, "rougeL": 1, "bleu1": 1, "bleu4": 1}} for task in data.TASKS]
    parent = copy.deepcopy(rows)
    next(row for row in rows if row["task_type"] == missing_task).update(filtered=True, accuracy=None)
    user["filter_policy"] = runtime.SPARSE_FILTER_PROTOCOL
    summary = runtime.summarize_user(user, rows)
    assert summary["accuracy_pct_micro"] == 100
    assert summary["task_scores"][missing_task] is None
    assert summary["missing_tasks"] == [missing_task] and summary["evaluable_tasks"] == 3
    paired = runtime.common_sample_scores(rows, parent, allow_missing_tasks=True)
    assert paired["left"] == paired["right"] == 100
    assert paired["missing_tasks"] == [missing_task]
    with pytest.raises(ValueError, match="Each MemoryCD task"):
        runtime.common_sample_scores(rows, parent)
    user["filter_policy"] = runtime.FILTER_PROTOCOL
    with pytest.raises(ValueError, match="Each MemoryCD task"):
        runtime.summarize_user(user, rows)
    user["filter_policy"] = runtime.SPARSE_FILTER_PROTOCOL
    for row in rows:
        row.update(filtered=True, accuracy=None)
    with pytest.raises(ValueError, match="Each MemoryCD user"):
        runtime.summarize_user(user, rows)
    with pytest.raises(ValueError, match="Each MemoryCD user"):
        runtime.common_sample_scores(rows, parent, allow_missing_tasks=True)
