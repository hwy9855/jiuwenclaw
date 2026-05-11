#!/usr/bin/env python3
"""
permute_context.py — Multi-hop QA document permutation prompt generator.

Based on: Huang et al., "Masking in Multi-hop QA: An Analysis of How Language
Models Perform with Context Permutation", ACL 2025.

Two modes:
  1. Generate mode  (default): reads a question+documents JSON file, outputs
     N prompt variants with different document orderings as JSON lines.
  2. Aggregate mode (--aggregate): reads a JSONL file of LLM answers from the
     N permutations and returns the majority-vote answer.

No external dependencies — pure Python stdlib.

Usage examples:
  # Generate 6 permutation prompts
  python permute_context.py --input question.json --n 6 --output prompts.jsonl

  # Generate with global-view prefix (Rule 4)
  python permute_context.py --input question.json --n 6 --global-prefix --output prompts.jsonl

  # Aggregate answers after calling LLM for each prompt
  python permute_context.py --aggregate answers.jsonl

Input file (question.json):
  {
    "question": "Who was president when the Eiffel Tower was built?",
    "documents": ["Doc text 1...", "Doc text 2...", "Doc text 3..."],
    "doc_titles": ["Title 1", "Title 2", "Title 3"]   // optional
  }

Output prompts.jsonl — one line per permutation:
  {"permutation_id": 0, "order": [0, 1, 2], "prompt": "..."}
  {"permutation_id": 1, "order": [1, 0, 2], "prompt": "..."}
  ...

Input answers.jsonl — one line per LLM response:
  {"permutation_id": 0, "answer": "Sadi Carnot"}
  {"permutation_id": 1, "answer": "Sadi Carnot"}
  {"permutation_id": 2, "answer": "Marie Curie"}
  ...
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from itertools import permutations


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_document_block(docs: list[str], titles: list[str], order: list[int]) -> str:
    lines = []
    for display_idx, doc_idx in enumerate(order, start=1):
        title = titles[doc_idx] if titles else f"Document {doc_idx + 1}"
        lines.append(f"Document {display_idx} — {title}:")
        lines.append(docs[doc_idx])
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_global_prefix(docs: list[str], titles: list[str], order: list[int]) -> str:
    lines = ["[CONTEXT OVERVIEW]"]
    lines.append(
        f"You will read {len(order)} documents to answer a multi-hop question. "
        "Brief overview of each:"
    )
    for display_idx, doc_idx in enumerate(order, start=1):
        title = titles[doc_idx] if titles else f"Document {doc_idx + 1}"
        # Use the first sentence of the document as a one-line summary.
        first_sentence = docs[doc_idx].split(".")[0].strip()
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."
        lines.append(f"  - Document {display_idx} ({title}): {first_sentence}.")
    lines.append("")
    lines.append("Keep this overview in mind as you read the full documents below.")
    lines.append("")
    return "\n".join(lines)


def build_prompt(
    question: str,
    docs: list[str],
    titles: list[str],
    order: list[int],
    use_global_prefix: bool = False,
) -> str:
    parts = []

    if use_global_prefix:
        parts.append(_build_global_prefix(docs, titles, order))
        parts.append("[FULL DOCUMENTS]\n")

    parts.append(_build_document_block(docs, titles, order))
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append(
        "Answer the question using only the information in the documents above. "
        "Think step by step: first identify the intermediate fact, then use it to "
        "reach the final answer. Be concise."
    )
    parts.append("")
    parts.append("Answer:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Permutation selection
# ---------------------------------------------------------------------------

def select_permutations(n_docs: int, n_max: int, seed: int = 42) -> list[list[int]]:
    """
    Return up to n_max distinct permutations of range(n_docs).

    Strategy (from the paper):
    - Always include the forward order [0, 1, ..., n-1] as permutation 0.
    - Always include the reverse order [n-1, ..., 1, 0] as permutation 1.
    - Fill remaining slots with random shuffles (deduplicated).
    """
    all_indices = list(range(n_docs))
    forward = list(all_indices)
    backward = list(reversed(all_indices))

    total_possible = math.factorial(n_docs)
    n_wanted = min(n_max, total_possible)

    if total_possible <= n_max:
        # Return all permutations, forward first
        result = [list(p) for p in permutations(all_indices)]
        result.remove(forward)
        if backward in result:
            result.remove(backward)
        return [forward, backward] + result if backward != forward else [forward] + result

    seen = {tuple(forward), tuple(backward)}
    selected = [forward, backward]

    rng = random.Random(seed)
    attempts = 0
    max_attempts = n_wanted * 20

    while len(selected) < n_wanted and attempts < max_attempts:
        candidate = list(all_indices)
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
        attempts += 1

    return selected[:n_wanted]


# ---------------------------------------------------------------------------
# Generate mode
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    question = data["question"]
    docs = data["documents"]
    titles = data.get("doc_titles", [])

    if not docs:
        print("ERROR: 'documents' list is empty.", file=sys.stderr)
        sys.exit(1)

    if titles and len(titles) != len(docs):
        print(
            f"WARNING: doc_titles length ({len(titles)}) != documents length ({len(docs)}). "
            "Ignoring titles.",
            file=sys.stderr,
        )
        titles = []

    permutation_orders = select_permutations(len(docs), args.n)

    output_lines = []
    for perm_id, order in enumerate(permutation_orders):
        prompt = build_prompt(
            question=question,
            docs=docs,
            titles=titles,
            order=order,
            use_global_prefix=args.global_prefix,
        )
        record = {
            "permutation_id": perm_id,
            "order": order,
            "prompt": prompt,
        }
        output_lines.append(json.dumps(record, ensure_ascii=False))

    output = "\n".join(output_lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {len(output_lines)} permutation prompts to {args.output}")
    else:
        sys.stdout.write(output)


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------

def aggregate(args: argparse.Namespace) -> None:
    answers = []
    with open(args.aggregate, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue
            answer = record.get("answer", "").strip()
            if answer:
                answers.append(answer)

    if not answers:
        print("ERROR: no answers found in the input file.", file=sys.stderr)
        sys.exit(1)

    # Normalise for counting: lowercase, strip punctuation from ends
    def normalise(text: str) -> str:
        return text.lower().strip(" .,;:!?\"'")

    normalised = [normalise(a) for a in answers]
    counter = Counter(normalised)
    winner_normalised, count = counter.most_common(1)[0]

    # Return the original-case version of the winning answer
    winner_original = next(
        a for a, n in zip(answers, normalised) if n == winner_normalised
    )

    result = {
        "majority_answer": winner_original,
        "vote_count": count,
        "total_answers": len(answers),
        "all_counts": dict(counter.most_common()),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-hop QA document permutation prompts (generate mode) or "
            "aggregate LLM answers via majority voting (aggregate mode)."
        )
    )

    # Generate mode args
    parser.add_argument(
        "--input",
        metavar="FILE",
        help=(
            'Path to input JSON file with "question" and "documents" keys. '
            "Required for generate mode."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        metavar="N",
        help="Maximum number of permutations to generate (default: 6).",
    )
    parser.add_argument(
        "--global-prefix",
        action="store_true",
        help=(
            "Prepend a global-view context overview before the documents "
            "(Rule 4: bi-directional attention workaround)."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Path to output JSONL file. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible permutation selection (default: 42).",
    )

    # Aggregate mode args
    parser.add_argument(
        "--aggregate",
        metavar="FILE",
        help=(
            'Path to JSONL file of LLM answers, each line: {"permutation_id": N, "answer": "..."}. '
            "Activates aggregate mode."
        ),
    )

    args = parser.parse_args()

    if args.aggregate:
        aggregate(args)
    elif args.input:
        generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
