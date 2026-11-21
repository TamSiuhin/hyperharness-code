"""MemoHarness-style successful/failed case retrieval with pinned Qwen embeddings."""
from collections import Counter
import hashlib
import json
import math
import os


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_INSTRUCTION = ("Given observed movie-tagging history, retrieve similar historical movie-tagging executions "
    "whose successes and failures can inform construction of this user's personalized harness.")


def policy():
    return {"method": "memoharness_qwen3_embedding_v1", "model": MODEL, "revision": REVISION,
        "backend": "vllm", "base_url": "http://127.0.0.1:8002/v1", "api_key_env": "QWEN_EMBEDDING_API_KEY",
        "device": "cuda", "dtype": "bfloat16", "timeout_seconds": 120, "batch_size": 8, "chunk_tokens": 512,
        "dimensions": 1024, "top_k_success": 5, "top_k_failure": 3, "query_instruction": QUERY_INSTRUCTION,
        "query": "Normalized mean of all known target-history record embeddings; no evaluation questions or labels.",
        "documents": "Historical prediction request only; answers, reward, diagnoses and later history excluded.",
        "ranking": "Cosine plus MemoHarness domain/external-knowledge/cluster and complexity/ambiguity/safety feature bonuses.",
        "long_text": "Token chunks, normalized per-text mean, then normalized history-record mean; no truncation.",
        "global_patterns": "Disabled: no LLM distillation and no skills.", "fallback": "Raise; never use another model or statistics silently."}


def normalized(matrix):
    import numpy as np
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embeddings cannot be ranked")
    return values / norms


def diverse_policy():
    return {**policy(), "retrieval_version": "cosine_diverse_v2", "feature_weight": 0.0,
        "max_per_donor": 2, "max_per_window": 1,
        "ranking": "Cosine only; at most one execution per S0 window and two per donor across both outcome pools. "
            "Caps are never relaxed; report any pool shortfall instead of duplicating cases."}


def label_quotas(events, available, count):
    counts = Counter(json.loads(event["text"])["tag"] for event in events)
    total = sum(counts[tag] for tag in available)
    if not total:
        return {}
    quotas = {tag: counts[tag] * count // total for tag in available if counts[tag]}
    remaining = count - sum(quotas.values())
    order = sorted(quotas, key=lambda tag: (-(counts[tag] * count % total), -counts[tag], tag))
    for tag in order[:remaining]:
        quotas[tag] += 1
    return {tag: quotas[tag] for tag in order if quotas[tag]}


class QwenEmbeddings:
    def __init__(self, configuration=None):
        self.configuration = configuration or policy()
        self.model = None
        self.calls = {"texts": 0, "chunks": 0, "tokens": 0}
        if self.configuration["model"] != MODEL or self.configuration["revision"] != REVISION:
            raise ValueError("Embedding model must match the pinned Qwen revision")

    def _load(self):
        backend = self.configuration.get("backend", "sentence_transformers")
        if backend == "vllm":
            self.model = VLLMEmbeddingModel(self.configuration)
            return
        if backend != "sentence_transformers":
            raise ValueError("Unsupported embedding backend; refusing fallback")
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(self.configuration["threads"])
        self.model = SentenceTransformer(MODEL, revision=REVISION, device=self.configuration["device"],
            trust_remote_code=False, model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "sdpa"})
        self.model.tokenizer.padding_side = "left"
        self.model.max_seq_length = self.configuration["chunk_tokens"] + 256
        self.model.eval()

    def encode(self, texts, *, query=False):
        import numpy as np
        if not texts:
            raise ValueError("Cannot embed empty text collection")
        if self.model is None:
            self._load()
        tokenizer = self.model.tokenizer
        chunks, spans = [], []
        prompt = f"Instruct: {self.configuration['query_instruction']}\nQuery: " if query else ""
        limit = self.configuration["chunk_tokens"]
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            if not tokens:
                raise ValueError("Cannot embed blank text")
            start = len(chunks)
            for offset in range(0, len(tokens), limit):
                chunk = tokenizer.decode(tokens[offset:offset + limit], skip_special_tokens=False)
                if len(tokenizer.encode(prompt + chunk, add_special_tokens=True)) > self.model.max_seq_length:
                    raise ValueError("Embedding chunk exceeds the registered context; refusing to truncate")
                chunks.append(chunk)
            spans.append((start, len(chunks)))
            self.calls["tokens"] += len(tokens)
        vectors = normalized(self.model.encode(chunks, prompt=prompt, batch_size=self.configuration["batch_size"],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        if vectors.shape[1] != self.configuration["dimensions"]:
            raise ValueError("Unexpected embedding dimension")
        self.calls["texts"] += len(texts)
        self.calls["chunks"] += len(chunks)
        return normalized(np.stack([vectors[start:end].mean(axis=0) for start, end in spans]))


class VLLMEmbeddingModel:
    def __init__(self, configuration, *, client=None, tokenizer=None):
        self.configuration = configuration
        self.max_seq_length = configuration["chunk_tokens"] + 256
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, padding_side="left")
        self.tokenizer = tokenizer
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=configuration["base_url"],
                api_key=os.environ.get(configuration["api_key_env"]) or "local-embedding",
                max_retries=0, timeout=configuration["timeout_seconds"])
        self.client = client
        if configuration["model"] not in {model.id for model in client.models.list().data}:
            raise ValueError("Requested embedding model is not served by vLLM")

    def encode(self, chunks, *, prompt, batch_size, **options):
        import numpy as np
        batches = []
        for start in range(0, len(chunks), batch_size):
            inputs = [prompt + chunk for chunk in chunks[start:start + batch_size]]
            response = self.client.embeddings.create(model=self.configuration["model"], input=inputs, encoding_format="float")
            if response.model != self.configuration["model"]:
                raise ValueError("Unexpected model in vLLM embedding response")
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(inputs))):
                raise ValueError("Missing or duplicate embedding response indices")
            vectors = normalized([row.embedding for row in rows])
            if vectors.shape[1] != self.configuration["dimensions"]:
                raise ValueError("Unexpected vLLM embedding dimension")
            batches.append(vectors)
        return np.concatenate(batches)


def history_texts(events):
    return ["Movie description: " + str(record["description"]) + "\nObserved user tag: " + str(record["tag"])
        for record in (json.loads(event["text"]) for event in events)]


def case_features(events, request=None):
    records = [json.loads(event["text"]) for event in events]
    if not records:
        raise ValueError("Case matching needs visible history")
    counts = Counter(record["tag"] for record in records)
    shares = [count / len(records) for count in counts.values()]
    entropy = -sum(share * math.log(share) for share in shares) / math.log(len(counts)) if len(counts) > 1 else 0.0
    length = len(request) if request is not None else sum(len("Movie description: " + str(record["description"])) for record in records) / len(records)
    return {"domain": "movie_tagging", "cluster_id": "movie_tagging", "requires_external_knowledge": False,
        "complexity_estimate": min(1.0, length / 2000), "ambiguity_score": entropy, "safety_sensitivity": 0.0}


def feature_bonus(left, right):
    score = 3.0 * (left["domain"] == right["domain"])
    score += 2.5 * (left["requires_external_knowledge"] == right["requires_external_knowledge"])
    score += float((left.get("cluster_id") or left["domain"]) == (right.get("cluster_id") or right["domain"]))
    return score + sum(max(0.0, 1.0 - abs(left[key] - right[key])) for key in
        ("complexity_estimate", "ambiguity_score", "safety_sensitivity"))


def prediction_cases(bank, assets):
    cases = []
    seen = set()
    for entry in bank["entries"]:
        timeline = [json.loads(line) for line in assets[entry["files"]["observation_timeline"]].splitlines()]
        for row in json.loads(assets[entry["files"]["history_predictions"]]):
            cutoff = row["history_cutoff"]
            if not entry["construction"]["history_events"] <= cutoff <= len(timeline):
                raise ValueError("Invalid visible-history cutoff in semantic case")
            if any(json.loads(event["text"])["date"] >= row["date"] for event in timeline[:cutoff]):
                raise ValueError("Semantic case contains same-day or future history")
            identity = entry["donor"], row["query_id"]
            if identity in seen:
                raise ValueError("Duplicate historical prediction in semantic index")
            seen.add(identity)
            case_id = entry["entry_id"] + "-" + hashlib.sha256(row["query_id"].encode()).hexdigest()[:16]
            cases.append({"case_id": case_id, "entry_id": entry["entry_id"], "donor": entry["donor"],
                "request": row["request"], "features": case_features(timeline[:cutoff], row["request"]),
                "success": row["accuracy"] == 100 and not row.get("runtime_violations"),
                "prediction": row})
    if not cases:
        raise ValueError("No actual history predictions to index")
    return cases


def build_index(bank, assets, embedder):
    cases = prediction_cases(bank, assets)
    vectors = normalized(embedder.encode([case["request"] for case in cases], query=False))
    if len(vectors) != len(cases):
        raise ValueError("Embedding count differs from prediction count")
    return cases, vectors


def retrieve(bank, cases, vectors, events, embedder, configuration=None, *, query_vector=None):
    import numpy as np
    configuration = configuration or policy()
    vectors = normalized(vectors)
    if len(vectors) != len(cases):
        raise ValueError("Semantic index row order does not match case manifest")
    texts = history_texts(events)
    if query_vector is None:
        history_vectors = normalized(embedder.encode(texts, query=True))
        if len(history_vectors) != len(texts) or history_vectors.shape[1] != vectors.shape[1]:
            raise ValueError("Query embeddings do not match semantic index")
        query = normalized(history_vectors.mean(axis=0, keepdims=True))[0]
    else:
        query = normalized([query_vector])[0]
        if query.shape[0] != vectors.shape[1]:
            raise ValueError("Cached query embedding dimension differs")
    features = case_features(events)
    cosines = vectors @ query
    feature_weight = configuration.get("feature_weight", 1.0)
    bonuses = [feature_weight * feature_bonus(features, case["features"]) for case in cases]
    ranked = sorted(range(len(cases)), key=lambda index: (
        -float(cosines[index]) - bonuses[index], cases[index]["case_id"]))
    groups, donor_counts, window_counts, coverage = {}, Counter(), Counter(), {}
    for group, success, limit in (("successful", True, configuration["top_k_success"]), ("failed", False, configuration["top_k_failure"])):
        groups[group] = []
        pool = [index for index in ranked if cases[index]["success"] == success]
        quotas = label_quotas(events, {cases[index]["prediction"].get("observed_tag") for index in pool}, limit) if configuration.get("label_coverage") else {}
        coverage[group] = {"requested_by_tag": quotas, "filled_by_tag": {}}
        candidates = []
        if quotas:
            for tag in quotas:
                candidates.extend(index for index in pool if cases[index]["prediction"].get("observed_tag") == tag)
        tag_counts = Counter()
        for index in candidates + pool:
            case = cases[index]
            if len(groups[group]) == limit:
                break
            if index in groups[group]:
                continue
            tag = case["prediction"].get("observed_tag")
            if index in candidates and tag_counts[tag] >= quotas[tag]:
                continue
            if donor_counts[case["donor"]] >= configuration.get("max_per_donor", len(cases)):
                continue
            if window_counts[case["entry_id"]] >= configuration.get("max_per_window", len(cases)):
                continue
            groups[group].append(index)
            donor_counts[case["donor"]] += 1
            window_counts[case["entry_id"]] += 1
            tag_counts[tag] += 1
        if quotas and len(groups[group]) < limit:
            for index in pool:
                case = cases[index]
                if len(groups[group]) == limit:
                    break
                if index in groups[group] or donor_counts[case["donor"]] >= configuration.get("max_per_donor", len(cases)) or window_counts[case["entry_id"]] >= configuration.get("max_per_window", len(cases)):
                    continue
                groups[group].append(index)
                donor_counts[case["donor"]] += 1
                window_counts[case["entry_id"]] += 1
                tag_counts[case["prediction"].get("observed_tag")] += 1
        coverage[group]["filled_by_tag"] = dict(tag_counts)
    entries = {entry["entry_id"]: entry for entry in bank["entries"]}
    selected = []
    for group, indices in groups.items():
        for index in indices:
            case = cases[index]
            parent = entries[case["entry_id"]]
            selected.append({"entry_id": case["case_id"], "parent_entry_id": case["entry_id"], "donor": case["donor"],
                "evidence": group, "historical_execution": case["prediction"], "files": parent["files"],
                "construction": parent["construction"], "window_performance": parent["performance"],
                "author_design_not_validated": parent["author_design_not_validated"]})
    return {"experience": {"origin": bank["origin"], "retrieval_method": configuration["method"],
        "interpretation": "Nearest successful and failed TRAIN executions, not causal revision gains. Their full S0 and "
            "history context are linked. Infer applicability from the target's own history; never copy donor answers.",
        "cases": selected},
        "audit": {"model": configuration["model"], "revision": configuration["revision"],
            "ranking_policy": {key: configuration.get(key) for key in ("retrieval_version", "feature_weight", "max_per_donor", "max_per_window")},
            "selected_counts": {group: len(indices) for group, indices in groups.items()},
            "distinct_donors": len(donor_counts), "distinct_windows": len(window_counts),
            "label_coverage": coverage,
            "query_history_events": len(events), "query_text_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
            "query_embedding": query.tolist(), "query_features": features,
            "requested": {"successful": configuration["top_k_success"], "failed": configuration["top_k_failure"]},
            "available": {"successful": sum(case["success"] for case in cases), "failed": sum(not case["success"] for case in cases)},
            "selected": [{"case_id": cases[index]["case_id"], "pool": group, "cosine": float(cosines[index]),
                "feature_bonus": bonuses[index],
                "score": float(cosines[index]) + bonuses[index]}
                for group, indices in groups.items() for index in indices],
            "used_evaluation_queries": False, "fallback": None}}
