"""Generate full versioned policy documents, then grounded evaluation questions.

Requires OPENAI_API_KEY. Nothing is generated until this script is run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path


MODEL = "gpt-4o-mini"
EVALUATION_DATE = date(2026, 10, 12)
EFFECTIVE_DATES = ("2024-01-01", "2025-01-01", "2026-01-01")
SECTION_NAMES = ("purpose", "scope", "requirements", "exceptions", "procedure", "records")
CHANGEABLE_SECTIONS = SECTION_NAMES[1:]
EXPLICIT_CHANGE_POLICIES = {"P03", "P06", "P09"}


POLICY_TOPICS = (
    {"id": "P01", "title": "Travel and Accommodation Policy", "focus": "business travel booking and reimbursement"},
    {"id": "P02", "title": "Expense Claims Policy", "focus": "submitting and approving employee expenses"},
    {"id": "P03", "title": "Procurement Approval Policy", "focus": "purchase approvals and spending authority"},
    {"id": "P04", "title": "Supplier Onboarding Policy", "focus": "checks before engaging a new supplier"},
    {"id": "P05", "title": "Security Incident Policy", "focus": "reporting and escalating suspected security incidents"},
    {"id": "P06", "title": "IT Access Review Policy", "focus": "granting and regularly reviewing system access"},
    {"id": "P07", "title": "Remote Work Policy", "focus": "eligibility, scheduling, and approval of remote work"},
    {"id": "P08", "title": "Records Retention Policy", "focus": "retention, access, and disposal of business records"},
    {"id": "P09", "title": "Employee Training Policy", "focus": "mandatory training and overdue training"},
    {"id": "P10", "title": "Contractor Site Access Policy", "focus": "contractor induction and site access"},
    {"id": "P11", "title": "Equipment Maintenance Policy", "focus": "inspection, maintenance, and reporting equipment faults"},
    {"id": "P12", "title": "Customer Complaint Policy", "focus": "acknowledging, escalating, and closing customer complaints"},
)


def object_schema(properties: dict) -> dict:
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


SECTIONS_SCHEMA = object_schema({name: {"type": "string"} for name in SECTION_NAMES})
QUESTION_SCHEMA = object_schema({
    "id": {"type": "string"},
    "question": {"type": "string"},
    "expected_answer": {"type": "string"},
    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
})
QUESTIONS_SCHEMA = object_schema({"questions": {"type": "array", "items": QUESTION_SCHEMA}})
FACT_TOKEN = re.compile(
    r"[$€£]\s?\d[\d,]*(?:\.\d+)?|"
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:business\s+)?(?:days?|hours?|months?|years?)\b",
    re.IGNORECASE,
)


def ask_json(client, model: str, name: str, schema: dict, instruction: str, material: dict) -> dict:
    """Ask the model for a JSON object matching a strict schema."""
    response = client.responses.create(
        model=model,
        temperature=0,
        instructions=instruction,
        input=json.dumps(material, ensure_ascii=False),
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
    )
    if not response.output_text:
        raise ValueError(f"No usable response for {name}")
    return json.loads(response.output_text)


def validate_sections(sections: dict) -> None:
    if set(sections) != set(SECTION_NAMES):
        raise ValueError("Policy must contain all six named sections")
    for name, text in sections.items():
        if not isinstance(text, str) or len(text.strip()) < 80:
            raise ValueError(f"Section {name} is too short for a useful policy document")


def create_v1(client, model: str, topic: dict) -> dict:
    sections = ask_json(
        client, model, "initial_policy", SECTIONS_SCHEMA,
        "Create an entirely fictional but realistic internal company policy. Write a complete document with six sections. "
        "Use several concrete, testable obligations across requirements, exceptions, procedure, and records: "
        "amounts, deadlines, approval roles, eligibility, or escalation rules where suitable. "
        "Each section must have at least two substantive sentences. Avoid real company names and legal claims. "
        "The document header supplies its version ID and effective-from date, so do not repeat those header fields "
        "inside the policy sections. Dates that are part of an actual rule or deadline are allowed.",
        topic,
    )
    validate_sections(sections)
    return sections


def reverted_facts(history: list[dict], updated: dict, targets: list[str]) -> list[str]:
    """Find amounts and durations restored from older versions in changed sections."""
    if len(history) < 2:
        return []
    previous = history[-1]
    restored = []
    for section in targets:
        previous_facts = {fact.casefold() for fact in FACT_TOKEN.findall(previous[section])}
        updated_facts = {fact.casefold() for fact in FACT_TOKEN.findall(updated[section])}
        for earlier in history[:-1]:
            earlier_facts = {fact.casefold() for fact in FACT_TOKEN.findall(earlier[section])}
            restored.extend(sorted((earlier_facts - previous_facts) & updated_facts))
    return sorted(set(restored))


def revise_policy(
    client, model: str, topic: dict, history: list[dict], next_version: str
) -> tuple[dict, list[dict]]:
    """Revise two selected sections, leaving others for the next version."""
    previous = history[-1]
    allowed_sections = list(CHANGEABLE_SECTIONS)
    if len(history) > 1:
        allowed_sections = [
            section for section in CHANGEABLE_SECTIONS
            if history[-2][section] == previous[section]
        ]
    if len(allowed_sections) < 2:
        raise ValueError(f"Not enough unchanged sections to revise {next_version}")
    offset = (int(topic["id"][1:]) - 1) % len(allowed_sections) if len(history) == 1 else 0
    targets = [allowed_sections[(offset + index) % len(allowed_sections)] for index in range(2)]
    target_schema = object_schema({section: {"type": "string"} for section in targets})
    problem = ""
    for _attempt in range(3):
        plan_instruction = (
            "Plan one material policy change for each requested section. "
            f"The sections are {', '.join(targets)}. Return one instruction string per section. "
            "Vary the kinds of changes: thresholds, deadlines, roles, exceptions, scope, or records. "
            "Use the latest version as the source of truth. Do not restore an earlier rule. "
            "Choose changes that do not require editing any other section. "
            "State old and new rules precisely."
        )
        if problem:
            plan_instruction += f" The previous attempt was rejected: {problem}"
        plan = ask_json(
            client, model, "revision_brief", target_schema,
            plan_instruction,
            {"policy": topic, "next_version": next_version, "earlier_versions": history[:-1],
             "current_sections": previous, "target_sections": targets},
        )
        brief = [{"section": section, "instruction": plan[section].strip()} for section in targets]
        if any(len(change["instruction"]) < 20 for change in brief):
            problem = "Each requested section needs a specific change instruction."
            continue

        replacements = ask_json(
            client, model, "revised_sections", target_schema,
            "Apply every change instruction. Return complete replacement text for each requested section. "
            "Keep all other rules in those sections unless explicitly changed. "
            "Keep repeated thresholds and deadlines consistent across the returned sections.",
            {"policy": topic, "current_sections": previous, "change_instructions": brief},
        )

        updated = previous.copy()
        for section in targets:
            updated[section] = replacements[section].strip()
        if any(updated[section] == previous[section] for section in targets):
            problem = "At least one requested section did not change."
            continue
        try:
            validate_sections(updated)
        except ValueError as exc:
            problem = str(exc)
            continue
        restored = reverted_facts(history, updated, targets)
        if restored:
            problem = f"It restored superseded values: {', '.join(restored)}. Choose a different change."
            continue
        return updated, brief
    raise ValueError(f"Could not create a coherent {next_version} for {topic['id']}: {problem}")


def version_dates(policy_number: int) -> tuple[str, ...]:
    # Every third policy has v1 (2024), v2 (2025), and v3 (2026).
    # The others have v1 (2024) and v2 (2026).
    return EFFECTIVE_DATES if policy_number % 3 == 0 else (EFFECTIVE_DATES[0], EFFECTIVE_DATES[2])


def render_document(
    topic: dict, version_id: str, effective_from: str, supersedes: str | None,
    sections: dict, revision_notes: list[dict] | None = None,
) -> str:
    lines = [
        f"# {topic['title']}", "",
        f"Policy ID: {topic['id']}",
        f"Version ID: {version_id}",
        f"Effective from: {effective_from}",
        f"Supersedes: {supersedes or 'None'}", "",
    ]
    for name in SECTION_NAMES:
        lines.extend((f"## {name.title()}", "", sections[name], ""))
    if revision_notes:
        lines.extend(("## Revision notes", "", f"Changes from {supersedes}:", ""))
        for change in revision_notes:
            lines.append(f"- {change['instruction']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def create_documents(
    client, model: str, selected_policy_id: str | None = None,
    documents: list[dict] | None = None, change_briefs: list[dict] | None = None,
    on_policy_complete=None,
) -> tuple[list[dict], list[dict]]:
    documents = list(documents or [])
    change_briefs = list(change_briefs or [])
    for number, topic in enumerate(POLICY_TOPICS, start=1):
        if selected_policy_id is not None and topic["id"] != selected_policy_id:
            continue
        existing = [doc for doc in documents if doc["policy_id"] == topic["id"]]
        if existing:
            if len(existing) != len(version_dates(number)):
                raise ValueError(f"Checkpoint has an incomplete {topic['id']}")
            print(f"Reusing completed {topic['id']} from checkpoint", flush=True)
            continue
        print(f"Creating {topic['id']}: {topic['title']}", flush=True)
        history = []
        dates = version_dates(number)
        for index, effective_from in enumerate(dates):
            version_id = f"{topic['id']}-v{index + 1}"
            supersedes = f"{topic['id']}-v{index}" if index else None
            brief = []
            if index == 0:
                sections = create_v1(client, model, topic)
            else:
                sections, brief = revise_policy(
                    client, model, topic, history, version_id
                )
                change_briefs.append({
                    "policy_id": topic["id"],
                    "from_version_id": supersedes,
                    "to_version_id": version_id,
                    "changes": brief,
                    "review_status": "pending_manual_review",
                })
            documents.append({
                "policy_id": topic["id"],
                "title": topic["title"],
                "version_id": version_id,
                "effective_from": effective_from,
                "effective_until": dates[index + 1] if index + 1 < len(dates) else None,
                "supersedes": supersedes,
                "sections": sections,
                "revision_changes": brief,
                "text": render_document(
                    topic, version_id, effective_from, supersedes, sections,
                    brief if topic["id"] in EXPLICIT_CHANGE_POLICIES else None,
                ),
            })
            history.append(sections)
        if on_policy_complete is not None:
            on_policy_complete(documents, change_briefs)
    return documents, change_briefs


def document_on(documents: list[dict], policy_id: str, when: str) -> dict:
    matches = [doc for doc in documents if doc["policy_id"] == policy_id and doc["effective_from"] <= when]
    if not matches:
        raise ValueError(f"No version of {policy_id} effective on {when}")
    return max(matches, key=lambda doc: doc["effective_from"])


def question_specs(documents: list[dict], as_of: date) -> list[dict]:
    """Choose 60 questions in the same six-category proportions as VersionQA."""
    specs = []

    def versions(policy_id: str) -> list[dict]:
        return [doc for doc in documents if doc["policy_id"] == policy_id]

    def stable_sections(policy_id: str) -> list[str]:
        policy_versions = versions(policy_id)
        return [
            section for section in CHANGEABLE_SECTIONS
            if all(doc["sections"][section] == policy_versions[0]["sections"][section]
                   for doc in policy_versions[1:])
        ]

    def changed_section(old: dict, new: dict) -> str:
        return next(
            section for section in CHANGEABLE_SECTIONS
            if old["sections"][section] != new["sections"][section]
        )

    def add(
        category: str, policy_id: str, when: str | None, required_versions: list[dict],
        evidence_sources: list[dict], evidence_sections: list[str], instruction: str,
    ) -> None:
        specs.append({
            "id": f"Q{len(specs) + 1:02d}", "category": category,
            "policy_id": policy_id, "as_of": when,
            "required_version_ids": [doc["version_id"] for doc in required_versions],
            "evidence_source_ids": [doc["version_id"] for doc in evidence_sources],
            "evidence_sections": evidence_sections,
            "instruction": instruction,
        })

    today = as_of.isoformat()
    # Basic and complex content questions use rules unchanged across all versions.
    for topic in POLICY_TOPICS:
        source = document_on(documents, topic["id"], today)
        section = stable_sections(topic["id"])[0]
        add("content_retrieval", topic["id"], None, [source], [source], [section],
            f"Ask a direct factual question answered by the unchanged {section} section. Do not mention a version.")

    for topic in POLICY_TOPICS:
        source = document_on(documents, topic["id"], today)
        stable = stable_sections(topic["id"])
        first = stable[0]
        second = stable[1] if len(stable) > 1 else stable[0]
        add("content_retrieval_complex", topic["id"], None, [source], [source, source],
            [first, second],
            f"Ask a question requiring synthesis of two different facts from the unchanged "
            f"{first} and {second} sections. If both section names are the same, use two different facts "
            "within that section. Give two distinct evidence quotes. Do not mention a version.")

    for topic in POLICY_TOPICS:
        when = "2025-06-01" if int(topic["id"][1:]) % 3 == 0 else "2024-06-01"
        source = document_on(documents, topic["id"], when)
        latest = versions(topic["id"])[-1]
        section = changed_section(source, latest)
        add("content_retrieval_version_specific", topic["id"], when, [source], [source], [section],
            f"Ask about a rule in the {section} section of {source['version_id']} on {when}. "
            "Include the version ID or date. The answer must differ from the latest version.")

    for topic in POLICY_TOPICS:
        policy_versions = versions(topic["id"])
        add("version_listing", topic["id"], None, policy_versions, policy_versions,
            ["header"] * len(policy_versions),
            "List every available version and its effective date.")

    for policy_id in ("P03", "P06", "P09"):
        old, new = versions(policy_id)[-2:]
        for change in new["revision_changes"][:2]:
            add("change_explicit", policy_id, None, [old, new], [new], ["revision_notes"],
                f"Ask what documented change in the {change['section']} section the revision notes "
                f"report from {old['version_id']} to {new['version_id']}. Quote that note as evidence.")

    for policy_id in ("P01", "P04", "P07"):
        old, new = versions(policy_id)[-2:]
        changed = [section for section in CHANGEABLE_SECTIONS
                   if old["sections"][section] != new["sections"][section]]
        for section in changed[:2]:
            add("change_implicit", policy_id, None, [old, new], [old, new], [section, section],
                f"Ask what changed in the {section} rule between {old['version_id']} and "
                f"{new['version_id']}. There are no revision notes; infer the change from both documents.")

    expected = {
        "content_retrieval": 12,
        "content_retrieval_complex": 12,
        "content_retrieval_version_specific": 12,
        "version_listing": 12,
        "change_explicit": 6,
        "change_implicit": 6,
    }
    if Counter(spec["category"] for spec in specs) != expected:
        raise ValueError("Question distribution is wrong")
    return specs


def create_questions(client, model: str, documents: list[dict], as_of: date) -> list[dict]:
    specs = question_specs(documents, as_of)
    questions = []
    for topic in POLICY_TOPICS:
        policy_specs = [spec for spec in specs if spec["policy_id"] == topic["id"]]
        policy_docs = [doc for doc in documents if doc["policy_id"] == topic["id"]]
        docs_by_id = {doc["version_id"]: doc for doc in policy_docs}
        for spec in policy_specs:
            if spec["category"] != "version_listing":
                continue
            versions = [docs_by_id[version_id] for version_id in spec["required_version_ids"]]
            questions.append({
                "id": spec["id"], "category": spec["category"],
                "policy_id": topic["id"], "as_of": None,
                "question": f"Which versions of the {topic['title']} are available, and when did each take effect?",
                "expected_answer": "; ".join(
                    f"{doc['version_id']} from {doc['effective_from']}" for doc in versions
                ) + ".",
                "required_version_ids": spec["required_version_ids"],
                "evidence_source_ids": spec["evidence_source_ids"],
                "evidence_quotes": [
                    f"Version ID: {doc['version_id']}\nEffective from: {doc['effective_from']}"
                    for doc in versions
                ],
                "review_status": "pending_manual_review",
            })
        model_specs = [spec for spec in policy_specs if spec["category"] != "version_listing"]
        if not model_specs:
            continue
        print(f"Creating questions for {topic['id']}", flush=True)
        generated = ask_json(
            client, model, "evaluation_questions", QUESTIONS_SCHEMA,
            "Write exactly one evaluation question and a concise correct answer for each supplied specification. "
            "Use only the supplied policy documents. Give one exact evidence quote for each evidence_source_id, "
            "in the specified order and from its corresponding evidence_section. The number of quotes may differ "
            "from the number of required_version_ids. For complex content questions, use two distinct facts "
            "and two distinct quotes from the specified section or sections. "
            "For explicit changes, quote revision notes. For implicit changes, compare the two version texts. "
            "Do not invent a policy rule or answer for a different date.",
            {"policy": topic, "specifications": model_specs,
             "documents": [{"version_id": doc["version_id"], "text": doc["text"]} for doc in policy_docs]},
        )["questions"]
        by_id = {item["id"]: item for item in generated}
        if len(generated) != len(model_specs) or set(by_id) != {spec["id"] for spec in model_specs}:
            raise ValueError(f"Missing or duplicate questions for {topic['id']}")
        for spec in model_specs:
            item = by_id[spec["id"]]
            quotes = item["evidence_quotes"]
            issues = []
            if not item["question"].strip() or not item["expected_answer"].strip():
                issues.append("Question or answer is empty")
            if len(quotes) != len(spec["evidence_source_ids"]):
                issues.append("Wrong number of evidence quotes")
            if spec["category"] == "content_retrieval_complex" and len(set(quotes)) != len(quotes):
                issues.append("Complex question repeats an evidence quote")
            for version_id, section, quote in zip(
                spec["evidence_source_ids"], spec["evidence_sections"], quotes
            ):
                source = docs_by_id[version_id]
                if section == "revision_notes":
                    source_text = source["text"].split("## Revision notes\n\n", 1)[1] if "## Revision notes\n\n" in source["text"] else ""
                elif section == "header":
                    source_text = source["text"].split("## Purpose", 1)[0]
                else:
                    source_text = source["sections"][section]
                if len(quote.strip()) < 12 or quote not in source_text:
                    issues.append(f"Evidence quote missing from {version_id}/{section}")
            questions.append({
                "id": spec["id"], "category": spec["category"],
                "policy_id": topic["id"], "as_of": spec["as_of"],
                "question": item["question"], "expected_answer": item["expected_answer"],
                "required_version_ids": spec["required_version_ids"],
                "evidence_source_ids": spec["evidence_source_ids"],
                "evidence_quotes": quotes,
                "review_status": "needs_manual_correction" if issues else "pending_manual_review",
                "validation_issues": issues,
            })
    return sorted(questions, key=lambda item: item["id"])


def render_changes(briefs: list[dict]) -> str:
    lines = ["# Policy version changes", ""]
    for brief in briefs:
        lines.extend((f"## {brief['from_version_id']} → {brief['to_version_id']}", ""))
        for change in brief["changes"]:
            lines.append(f"- **{change['section'].title()}:** {change['instruction']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_documents(output: Path, documents: list[dict], briefs: list[dict]) -> None:
    """Save complete Markdown documents and a readable change log."""
    (output / "policies").mkdir(parents=True, exist_ok=True)
    for doc in documents:
        (output / "policies" / f"{doc['version_id']}.md").write_text(doc["text"], encoding="utf-8")
    (output / "changes.md").write_text(render_changes(briefs), encoding="utf-8")


def save_dataset(output: Path, documents: list[dict], briefs: list[dict], questions: list[dict]) -> None:
    save_documents(output, documents, briefs)
    (output / "questions.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, documents: list[dict], briefs: list[dict]) -> None:
    """Keep completed policy families so a failed run can resume without regenerating them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({"documents": documents, "briefs": briefs}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=EVALUATION_DATE,
                        help="Fixed date for current-policy questions (default: 2026-10-12)")
    parser.add_argument("--model", default=MODEL, help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--policy", choices=[topic["id"] for topic in POLICY_TOPICS],
                        help="Preview one policy family and its versions, without generating questions")
    parser.add_argument("--output", type=Path, help="Output directory (default: dataset or dataset/preview-P03)")
    parser.add_argument("--overwrite", action="store_true", help="Replace a previous generated dataset")
    args = parser.parse_args()
    output = args.output or (Path("dataset") / f"preview-{args.policy}" if args.policy else Path("dataset"))

    if args.as_of < date(2024, 1, 1):
        parser.error("--as-of must be on or after 2024-01-01")
    first_preview_file = output / "policies" / f"{args.policy}-v1.md" if args.policy else None
    first_full_file = output / "policies" / "P01-v1.md"
    checkpoint_path = output / ".generation_checkpoint.json"
    if not args.overwrite and (first_full_file.exists() or (first_preview_file and first_preview_file.exists())):
        parser.error("Output already exists. Use --overwrite to replace it.")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY before running this script")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install the dependency with: python3 -m pip install -r requirements.txt")

    client = OpenAI()
    if args.policy:
        documents, briefs = create_documents(client, args.model, args.policy)
        save_documents(output, documents, briefs)
        print(f"Created {len(documents)} versions and a change summary in {output}")
        print("Please review the versions and changes before using them as ground truth")
    else:
        if args.overwrite and checkpoint_path.exists():
            checkpoint_path.unlink()
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
        documents, briefs = create_documents(
            client, args.model, documents=checkpoint.get("documents"),
            change_briefs=checkpoint.get("briefs"),
            on_policy_complete=lambda docs, changes: save_checkpoint(checkpoint_path, docs, changes),
        )
        questions = create_questions(client, args.model, documents, args.as_of)
        save_dataset(output, documents, briefs, questions)
        checkpoint_path.unlink(missing_ok=True)
        print(f"Created {len(documents)} versions and {len(questions)} questions in {output}")
        print("Please review everything before using them as ground truth")


if __name__ == "__main__":
    main()
