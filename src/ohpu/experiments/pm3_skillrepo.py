"""Versioned, TRAIN-curated Markdown skills for the frozen harness generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


VERSION = "skillrepo-v1"
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

MINIMAL_SKILL_NAME = "history-to-harness"
MINIMAL_SKILL_PATH = Path(__file__).parent / "skills" / MINIMAL_SKILL_NAME / "SKILL.md"
MINIMAL_SKILL_TEXT = MINIMAL_SKILL_PATH.read_text(encoding="utf-8")

CURATION_PROMPT = """
Maintain the shared construction SkillRepo, not a monolithic SKILL.md or a shared executable harness.
Read skill/repo.json and its skills/*.md entries. The initial repo contains a minimal history-to-harness skill. Each skill is a self-contained Markdown
procedure with YAML name and description describing when it applies. Learn reusable construction methods from TRAIN
experience, never donor preferences, identities or answers. Skills are consumed by the frozen harness generator,
not injected into the answering model. Do not require arbitrary per-user code diversity.
Submit edits/<candidate>.json with parent_sha256 and operations. Each operation has op, name, reason and evidence;
evidence is a nonempty list of user/query_id objects from the supplied TRAIN labels, kept only in the outer audit.
insert_skill also has description and content (Markdown body only). It creates a new named skill.
update_skill has patches, a list of old/new string objects; each old text must match exactly once in the current body.
It may also have description to improve discovery. Keep unrelated text intact. Apply patches in order.
delete_skill removes an existing harmful, redundant or obsolete skill. Do not delete/reinsert a skill to hide a rewrite.
Use lowercase hyphenated names. Do not submit a replacement repo or edit parent files directly. No scripts, assets or
local file links are allowed inside entries; each procedure must stand alone. Skill and repository lengths, skill counts,
operation counts and edit sizes are unlimited. There is no length reward or penalty. Host runtime limits still apply.
Write hypotheses/<candidate>.json with mechanism, expected_effect, regression_risks, target_tasks, risk_tasks and
consumer_hooks. Use retrieval receipts to diagnose whether a useful skill was surfaced; retrieval is not proof of use
or causal benefit. Only the candidate repo that passes the existing TRAIN gate becomes the next parent.
"""


def document(name, description, content):
    return "---\n" + yaml.safe_dump({"name": name, "description": description}, sort_keys=False, allow_unicode=True) + "---\n" + content


def parse_skill(name, text):
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ValueError("skill name must be lowercase and hyphenated")
    if not isinstance(text, str) or "\x00" in text:
        raise ValueError("skill must be Markdown text without NUL bytes")
    match = re.fullmatch(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", text, re.DOTALL)
    if match is None or not match.group(2).strip():
        raise ValueError("skill requires YAML frontmatter and a nonempty procedure")
    try:
        node = yaml.compose(match.group(1), Loader=yaml.SafeLoader)
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise ValueError("invalid skill YAML") from error
    if (not isinstance(node, yaml.MappingNode) or len(node.value) != 2 or not isinstance(metadata, dict)
            or set(metadata) != {"name", "description"} or metadata["name"] != name
            or not isinstance(metadata["description"], str) or not metadata["description"].strip()):
        raise ValueError("skill frontmatter requires matching name and a nonempty description")
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", match.group(2)):
        if not target.startswith(("#", "https://", "http://")):
            raise ValueError("SkillRepo entries are self-contained; local file links are not allowed")
    return metadata, match.group(2)


class SkillRepo:
    def __init__(self, skills=None):
        if skills is not None and not isinstance(skills, dict):
            raise ValueError("skills must be a name-to-Markdown mapping")
        self.skills = dict(skills or {})
        for name, text in self.skills.items():
            parse_skill(name, text)

    @classmethod
    def minimal(cls):
        return cls({MINIMAL_SKILL_NAME: MINIMAL_SKILL_TEXT})

    @classmethod
    def from_files(cls, files):
        if not isinstance(files, dict) or "repo.json" not in files:
            raise ValueError("SkillRepo requires repo.json")
        try:
            manifest = json.loads(files["repo.json"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid SkillRepo manifest") from error
        if (not isinstance(manifest, dict) or set(manifest) != {"version", "skills"} or manifest["version"] != VERSION
                or not isinstance(manifest["skills"], list) or any(not isinstance(name, str)
                    or NAME.fullmatch(name) is None for name in manifest["skills"])
                or len(set(manifest["skills"])) != len(manifest["skills"])):
            raise ValueError("invalid SkillRepo manifest")
        names = manifest["skills"]
        if set(files) != {"repo.json", *(f"skills/{name}.md" for name in names)}:
            raise ValueError("SkillRepo files must exactly match the manifest")
        return cls({name: files[f"skills/{name}.md"] for name in names})

    def files(self):
        return {"repo.json": json.dumps({"version": VERSION, "skills": sorted(self.skills)}, indent=2) + "\n",
            **{f"skills/{name}.md": self.skills[name] for name in sorted(self.skills)}}

    def sha256(self):
        return hashlib.sha256(json.dumps(self.files(), sort_keys=True).encode()).hexdigest()

    def retrieve(self, query, *, top_k=5):
        from ohpu.benchmarks.retrieval import bm25_scores

        if not isinstance(query, str) or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("retrieval requires text and a positive top_k")
        names = sorted(self.skills)
        texts = []
        for name in names:
            metadata, content = parse_skill(name, self.skills[name])
            texts.append(name + "\n" + metadata["description"] + "\n" + content)
        scores = bm25_scores(query, texts)
        ranked = sorted(range(len(names)), key=lambda index: (-scores[index], names[index]))
        return [{"name": names[index], "score": scores[index],
            "sha256": hashlib.sha256(self.skills[names[index]].encode()).hexdigest(),
            "path": f".agents/skills/{names[index]}/SKILL.md"} for index in ranked[:top_k] if scores[index] > 0]

    def validate_labels(self, labels):
        for text in self.skills.values():
            normalized = " ".join(text.split()).casefold()
            if "reference_harness.py" in text or "## Verified reference harness" in text:
                raise ValueError("SkillRepo cannot inherit a shared reference")
            for label in labels:
                for key in ("request", "gold_response", "inferior_response"):
                    value = " ".join(str(label.get(key) or "").split()).casefold()
                    if len(value) >= 40 and value in normalized:
                        raise ValueError("skill copies TRAIN label text")
        return self

    def apply(self, proposal, labels):
        if not isinstance(proposal, dict) or set(proposal) != {"parent_sha256", "operations"}:
            raise ValueError("curation requires parent_sha256 and operations, not a replacement repo")
        if proposal["parent_sha256"] != self.sha256():
            raise ValueError("curation refers to a different parent repo")
        operations = proposal["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("curation requires nonempty operations")
        known = {(str(row["user"]), row["query_id"]) for row in labels}
        updated, touched = dict(self.skills), set()
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("each curation operation must be an object")
            action, name = operation.get("op"), operation.get("name")
            if action not in ("insert_skill", "update_skill", "delete_skill"):
                raise ValueError("unknown skill operation")
            if not isinstance(name, str) or NAME.fullmatch(name) is None:
                raise ValueError("invalid skill name")
            if name in touched:
                raise ValueError("edit each skill only once per candidate")
            touched.add(name)
            required = {"op", "name", "reason", "evidence"}
            if action == "insert_skill":
                required |= {"description", "content"}
            elif action == "update_skill":
                required |= {"patches"}
            allowed = required | ({"description"} if action == "update_skill" else set())
            if not required <= set(operation) or set(operation) - allowed:
                raise ValueError("invalid skill operation fields")
            if not isinstance(operation["reason"], str) or not operation["reason"].strip():
                raise ValueError("curation requires a reason")
            evidence = operation["evidence"]
            if not isinstance(evidence, list) or not evidence or any(not isinstance(item, dict)
                    or set(item) != {"user", "query_id"} or not isinstance(item["user"], str)
                    or not isinstance(item["query_id"], str) or (item["user"], item["query_id"]) not in known for item in evidence):
                raise ValueError("curation requires known TRAIN evidence")
            if (action == "insert_skill") == (name in updated):
                raise ValueError("insert needs a new skill; update/delete need an existing skill")
            if action == "delete_skill":
                del updated[name]
            elif action == "insert_skill":
                if not isinstance(operation["content"], str):
                    raise ValueError("skill content must be text")
                updated[name] = document(name, operation["description"], operation["content"])
            else:
                metadata, content = parse_skill(name, updated[name])
                patches = operation["patches"]
                if not isinstance(patches, list) or (not patches and "description" not in operation):
                    raise ValueError("update requires patches or a description change")
                for patch in patches:
                    if (not isinstance(patch, dict) or set(patch) != {"old", "new"} or not isinstance(patch["old"], str)
                            or not patch["old"] or not isinstance(patch["new"], str) or patch["old"] == patch["new"]
                            or content.count(patch["old"]) != 1):
                        raise ValueError("each patch must replace one unambiguous body span")
                    content = content.replace(patch["old"], patch["new"], 1)
                updated[name] = document(name, operation.get("description", metadata["description"]), content)
                if updated[name] == self.skills[name]:
                    raise ValueError("skill update did not change anything")
        return SkillRepo(updated).validate_labels(labels)
