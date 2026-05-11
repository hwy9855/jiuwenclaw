---
name: multihop-qa-reasoning
description: >-
  Apply research-backed prompt engineering rules for Multi-hop Question Answering (MHQA) over retrieved documents.
  Encodes findings from "Masking in Multi-hop QA" (Huang et al., ACL 2025): document ordering, proximity,
  permutation sampling for self-consistency, global-view prefix as bi-directional attention workaround, and
  first-hop priority under truncation. Use when the user asks a multi-hop question over multiple documents,
  wants to improve RAG accuracy on complex reasoning tasks, mentions context ordering, lost-in-the-middle,
  multi-hop reasoning, or asks how to structure retrieved documents for LLM prompts.
---

# Multi-hop QA Reasoning

Prompt engineering rules for answering multi-hop questions over retrieved documents, based on Huang et al. (ACL 2025).

## The 5 Rules

### Rule 1 — Forward Document Order
**Always place documents in reasoning-chain order.**

If a question requires two hops (Doc A establishes an intermediate fact, Doc B uses that fact to reach the answer), put Doc A before Doc B. Reverse order degrades performance significantly for decoder-only models.

```
WRONG:  [Doc B (hop 2), irrelevant docs..., Doc A (hop 1)]
RIGHT:  [Doc A (hop 1), Doc B (hop 2), irrelevant docs...]
```

How to determine order: ask the LLM to decompose the question into sub-questions, then rank documents by which sub-question they answer.

### Rule 2 — Proximity Rule
**Keep relevant documents adjacent. Do not bury them under irrelevant text.**

Large gaps between gold documents ("lost in between") hurt performance. When constructing the prompt context, cluster the relevant documents together, with irrelevant docs placed at the end.

```
WRONG:  [Doc A, 5 irrelevant docs, Doc B, 3 more irrelevant docs]
RIGHT:  [Doc A, Doc B, irrelevant docs...]
```

### Rule 3 — Permutation Sampling (Self-Consistency Proxy)
**When uncertain, sample multiple document orderings and majority-vote the answer.**

The paper shows LMs assign higher attention peaks to relevant documents when they answer correctly. Since API access gives no attention weights, simulate this via self-consistency: generate answers from N permutations and keep the most frequent answer.

Workflow:
1. Use `scripts/permute_context.py` to generate N prompt variants (default: 6 permutations)
2. Send each prompt to the LLM
3. Run `--aggregate` mode on the responses to get majority-vote answer

This replicated the paper's heuristic improvement (Qwen 7B: 28.6% → 33.7% accuracy).

See [reference.md](reference.md) for the full prompt template used in permutation sampling.

### Rule 4 — Global-View Prefix (Bi-directional Attention Workaround)
**Prepend a brief summary of all documents before the individual documents.**

Decoder-only models (Qwen, Llama) use a causal mask and cannot attend backwards. A global-view prefix gives the model a forward pass over all content before it reads each document in detail, partially compensating for this limitation.

Prompt structure:
```
[CONTEXT OVERVIEW]
The following documents cover: {one-sentence summary of each doc}

[DOCUMENTS]
Document 1: {full text}
Document 2: {full text}
...

[QUESTION]
{question}
```

### Rule 5 — First-Hop Priority Under Truncation
**When context must be shortened, always keep the first-hop document.**

Removing the first-hop document (the one that establishes the intermediate entity) hurts more than removing the second-hop document. If token limits force you to drop a document, drop the last-hop document first.

Priority order when truncating: keep hop-1 doc > keep hop-2 doc > keep other relevant docs > drop irrelevant docs

---

## MHQA Workflow

```
Task Progress:
- [ ] Step 1: Decompose the question into sub-questions
- [ ] Step 2: Match each document to the sub-question it answers
- [ ] Step 3: Sort documents in forward reasoning order (Rule 1)
- [ ] Step 4: Cluster relevant docs together (Rule 2)
- [ ] Step 5: Add global-view prefix (Rule 4)
- [ ] Step 6: If confidence is low, run permutation sampling (Rule 3)
```

### Step 1 — Decompose the question

Use this prompt to get the reasoning chain:

```
Decompose the following multi-hop question into ordered sub-questions.
For each sub-question, identify which document (by title or key phrase) answers it.

Question: {question}
Documents: {list of document titles or first sentences}

Output format:
Sub-question 1: ...  →  Answered by: Document X
Sub-question 2: ...  →  Answered by: Document Y
```

### Step 2–4 — Order and cluster documents

Once you have the sub-question-to-document mapping, reorder the document list so:
- Document answering sub-question 1 comes first
- Document answering sub-question 2 comes second
- All remaining documents follow

### Step 5 — Full MHQA prompt with global-view prefix

```
[CONTEXT OVERVIEW]
You are given {N} documents. Summary: {doc1_title} covers {one_sentence}. {doc2_title} covers {one_sentence}. ...

[DOCUMENTS]
Document 1 — {title}:
{full_text}

Document 2 — {title}:
{full_text}

...

[QUESTION]
{question}

Answer the question using only the information in the documents above.
Think step by step: first identify the intermediate fact from Document 1, then use it with Document 2 to reach the final answer.
```

---

## Scripts

### `scripts/permute_context.py`

Generates N document permutation prompts for self-consistency voting. Pure Python stdlib, no model calls.

**Generate permutations:**
```bash
python scripts/permute_context.py --input question.json --n 6 --output prompts.jsonl
```

Input `question.json` format:
```json
{
  "question": "Who was the president when the Eiffel Tower was built?",
  "documents": ["Doc text 1...", "Doc text 2...", "Doc text 3..."],
  "global_view_prefix": true
}
```

Output `prompts.jsonl`: one JSON object per line, each with `permutation_id` and `prompt`.

**Aggregate answers after calling the LLM:**
```bash
python scripts/permute_context.py --aggregate answers.jsonl
```

Input `answers.jsonl` format: one JSON object per line with `{"permutation_id": 0, "answer": "..."}`.

---

## Additional Resources

- Full paper results, architecture comparison tables, and expanded prompt templates: [reference.md](reference.md)
- Paper: Huang et al., "Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation", ACL 2025. Code: https://github.com/hwy9855/MultiHopQA-Reasoning
