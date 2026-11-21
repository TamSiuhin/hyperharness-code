"""Sample long-history HORIZON users from compact, pinned Amazon ID tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from ohpu.benchmarks import horizon


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return json.load(response)


def source_manifest():
    url = f"https://raw.githubusercontent.com/microsoft/horizon-benchmark/{horizon.HORIZON_REVISION}/data/category_names.txt"
    with urllib.request.urlopen(url, timeout=60) as response:
        categories = response.read().decode().splitlines()
    categories = sorted(category.strip() for category in categories if category.strip())
    base = f"https://huggingface.co/api/datasets/{horizon.AMAZON_REPOSITORY}/tree/{horizon.AMAZON_REVISION}"
    entries = fetch_json(base + "/benchmark/0core/rating_only?recursive=false&expand=false")
    selected = [entry for entry in entries if Path(entry["path"]).stem in categories]
    selected += [entry for entry in fetch_json(base + "?recursive=false&expand=false") if entry["path"] == "asin2category.json"]
    assert len(selected) == len(categories) + 1
    return {"amazon_revision": horizon.AMAZON_REVISION, "horizon_revision": horizon.HORIZON_REVISION,
        "categories": categories, "files": [{"path": entry["path"], "bytes": entry["size"],
            "sha256": entry["lfs"]["oid"]} for entry in selected]}


def download_file(entry, cache):
    path = cache / Path(entry["path"]).name
    if path.exists():
        if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
            raise ValueError(f"Cached source checksum mismatch: {path.name}")
        return path
    url = f"https://huggingface.co/datasets/{horizon.AMAZON_REPOSITORY}/resolve/{horizon.AMAZON_REVISION}/{entry['path']}"
    partial = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with partial.open("xb") as target:
        with urllib.request.urlopen(url, timeout=180) as response:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
    if partial.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Downloaded source checksum mismatch: {path.name}")
    partial.rename(path)
    print(json.dumps({"downloaded": path.name, "bytes": entry["bytes"]}), flush=True)
    return path
