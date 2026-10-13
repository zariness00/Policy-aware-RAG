"""Run one policy QA pipeline and save both pipelines' answers in one CSV.

Use --dry-run to inspect the selected questions without importing a retriever,
calling an API, or writing an output file.
"""

import argparse
import csv
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
QUESTIONS_CSV = ROOT / "dataset" / "questions_final.csv"
VERSIONRAG_SRC = ROOT / "versionrag" / "src"

FIELDS = [
    "Type",
    "Question",
    "Answer",
    "Response_Baseline",
    "Response_VersionRAG",
]

RESPONSE_FIELDS = {
    "baseline": "Response_Baseline",
    "versionrag": "Response_VersionRAG",
}


def load_questions():
    with QUESTIONS_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Type", "Question", "Answer"]:
            raise ValueError("questions_final.csv must have Type,Question,Answer columns")
        public_questions = list(reader)

    questions = []
    for number, row in enumerate(public_questions, start=1):
        if not row["Question"] or not row["Answer"]:
            raise ValueError(f"Missing question or answer in CSV row {number}")
        questions.append({
            "id": f"Q{number:02d}",
            "type": row["Type"],
            "question": row["Question"],
            "answer": row["Answer"],
        })

    if len({item["question"] for item in questions}) != len(questions):
        raise ValueError("Duplicate questions in questions_final.csv")
    return questions


def load_results(output_path, questions):
    rows = [
        {
            "Type": question["type"],
            "Question": question["question"],
            "Answer": question["answer"],
            "Response_Baseline": "",
            "Response_VersionRAG": "",
        }
        for question in questions
    ]
    if not output_path.exists():
        return rows

    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected columns in {output_path}")
        saved_rows = list(reader)
    if len(saved_rows) != len(rows):
        raise ValueError(f"Unexpected number of questions in {output_path}")
    for expected, saved in zip(rows, saved_rows):
        if any(expected[field] != saved[field] for field in ("Type", "Question", "Answer")):
            raise ValueError(f"Questions in {output_path} differ from questions_final.csv")
        for field in RESPONSE_FIELDS.values():
            expected[field] = saved[field]
    return rows


def save_results(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def make_pipeline(pipeline):
    # The cloned repo uses relative paths for its Milvus database.
    os.chdir(VERSIONRAG_SRC)
    sys.path.insert(0, str(VERSIONRAG_SRC))

    if pipeline == "baseline":
        from retrieval.baseline_retriever import BaselineRetriever
        from generation.baseline_generator import BaselineGenerator

        return BaselineRetriever(), BaselineGenerator()

    from retrieval.versionrag_retriever import VersionRAGRetriever
    from generation.versionrag_generator import VersionRAGGenerator

    return VersionRAGRetriever(), VersionRAGGenerator()


def retrieve(pipeline, retriever, question):
    if pipeline == "baseline":
        return "SemanticSearch", {}, retriever.retrieve(question)

    # This is the same parser-then-database path used by retriever.retrieve().
    # Calling it here exposes the chosen mode without a second LLM call.
    parameters = retriever.parser.parse_retrieval_mode(question)
    evidence = retriever.database.retrieve(parameters)
    return parameters.retrieval_type.name, dict(parameters.params), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("baseline", "versionrag"), required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=1,
                           help="Use the first N questions; defaults to 1. Use 60 for the full set.")
    selection.add_argument("--ids", nargs="+", metavar="QNN",
                           help="Run only the listed question IDs, such as Q15 Q43 Q50 Q60.")
    parser.add_argument("--output", type=Path,
                        help="Output CSV path; defaults to dataset/evaluation_results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending questions; make no API calls or file changes")
    args = parser.parse_args()

    questions = load_questions()
    if args.ids:
        index_by_id = {question["id"]: index for index, question in enumerate(questions)}
        unknown = [question_id for question_id in args.ids if question_id not in index_by_id]
        if unknown:
            parser.error(f"Unknown question IDs: {', '.join(unknown)}")
        if len(set(args.ids)) != len(args.ids):
            parser.error("Question IDs must not be repeated")
        selected = [index_by_id[question_id] for question_id in args.ids]
    else:
        if not 1 <= args.limit <= len(questions):
            parser.error(f"--limit must be between 1 and {len(questions)}")
        selected = list(range(args.limit))

    output_path = args.output or ROOT / "dataset" / "evaluation_results.csv"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    rows = load_results(output_path, questions)
    response_field = RESPONSE_FIELDS[args.pipeline]
    pending = [index for index in selected if not rows[index][response_field].strip()]

    print(f"Pipeline: {args.pipeline}")
    print(f"Selected: {len(selected)}; already saved: {len(selected) - len(pending)}; pending: {len(pending)}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("Pending IDs:", ", ".join(questions[index]["id"] for index in pending) or "none")
        return
    if not pending:
        return

    retriever, generator = make_pipeline(args.pipeline)
    for index in pending:
        question = questions[index]
        print(f"{question['id']}: {question['question']}", flush=True)
        mode, _, evidence = retrieve(args.pipeline, retriever, question["question"])
        rows[index][response_field] = generator.generate(evidence, question["question"]).answer
        save_results(output_path, rows)
        print(f"Saved {question['id']} (mode: {mode})", flush=True)


if __name__ == "__main__":
    main()
