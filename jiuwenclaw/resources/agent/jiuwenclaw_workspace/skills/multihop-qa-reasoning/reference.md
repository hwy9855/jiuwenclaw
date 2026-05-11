# Multi-hop QA Reasoning — Reference

Source paper: **Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation**
Wenyu Huang, Pavlos Vougiouklis, Mirella Lapata, Jeff Z. Pan — ACL 2025
Code: https://github.com/hwy9855/MultiHopQA-Reasoning

---

## Paper Overview

The paper conducts a systematic study of how Language Models (LMs) answer multi-hop questions (MHQA) when retrieved documents are presented in different orders (permutations). The core finding is that **document order has a large, measurable impact** on MHQA accuracy.

---

## Key Results

### Document Order Impact

Three permutation conditions studied:
- **Forward**: Doc answering hop 1 appears before doc answering hop 2 → **best performance**
- **Backward**: Reversed → significantly degraded
- **Random**: Mixed → intermediate, unpredictable

Fine-tuned models develop a strong forward-order bias. Forward order is robustly the best configuration across all model families tested.

### Distance Between Gold Documents

Gold documents placed far apart (many irrelevant documents in between) degrade performance. The "lost in between" effect (analogous to "lost in the middle") is a separate failure mode from ordering.

| Distance (irrelevant docs between gold docs) | Accuracy |
|------|---------|
| Adjacent (0 irrelevant between) | Highest |
| 3–5 irrelevant docs between | Moderate drop |
| 7+ irrelevant docs between | Significant drop |

### Completeness: Removing Gold Documents

| Condition | Accuracy |
|-----------|---------|
| Both gold docs present | Baseline |
| First-hop doc removed | Large drop |
| Second-hop doc removed | Smaller drop |

**Takeaway**: The first-hop document (which establishes the intermediate entity) is more critical than the second-hop document.

### Attention-Peak Heuristic

The paper shows that LMs assign **higher peak attention weights** to at least one document when they answer correctly versus incorrectly. This is a signal of model confidence.

Since API access does not expose attention weights, the practical equivalent is **self-consistency permutation sampling**:
- Generate answers from N different document permutations
- The permutation where the model "concentrates" attention on the right document is likely to produce the correct answer
- Majority voting across permutations approximates selecting the highest-attention answer

**Measured improvement** (Qwen 7B): 28.6% → 33.7% accuracy using 6-permutation sampling.

### Global-View Prefix

Prepending a brief summary of all documents before the full document texts improves cross-document reasoning. It forces the model to process a high-level overview before attending to each document in sequence, increasing robustness when document order is suboptimal.

---

## Expanded Prompt Templates

### Template A — Standard MHQA with Forward Order

Use when you have identified the correct document order:

```
You are given a set of documents retrieved for a multi-hop question.
Read all documents carefully, then answer the question step by step.

Document 1:
{hop_1_document_text}

Document 2:
{hop_2_document_text}

{additional_documents_if_any}

Question: {question}

Instructions:
1. Identify the intermediate fact from Document 1 that connects to the question.
2. Use that fact together with Document 2 to reach the final answer.
3. State your answer clearly.

Answer:
```

### Template B — Global-View Prefix (Rule 4)

Use when document order is uncertain or when reasoning across many documents:

```
[CONTEXT OVERVIEW]
You will be given {N} documents to answer a multi-hop question.
Here is a brief overview of each document:
- Document 1 ({title_or_key_phrase}): {one_sentence_summary}
- Document 2 ({title_or_key_phrase}): {one_sentence_summary}
- Document 3 ({title_or_key_phrase}): {one_sentence_summary}

Keep this overview in mind as you read the full documents below.

[FULL DOCUMENTS]

Document 1 — {title}:
{full_text}

Document 2 — {title}:
{full_text}

Document 3 — {title}:
{full_text}

[QUESTION]
{question}

Think step by step. First identify which documents are relevant to each part of the question,
then combine the information to produce your final answer.

Answer:
```

### Template C — Permutation Sampling (Rule 3)

This is the system prompt used for each permutation variant. The document order changes per permutation; everything else stays the same.

```
You are answering a multi-hop question using retrieved documents.
Read the documents carefully and answer directly. Be concise.

{permuted_document_block}

Question: {question}

Answer (one sentence or short phrase):
```

After collecting N answers, aggregate with:
```
python scripts/permute_context.py --aggregate answers.jsonl
```

### Template D — Question Decomposition

Use before ordering documents (Step 1 of the workflow):

```
Break the following multi-hop question into a chain of simpler sub-questions.
For each sub-question, write which type of document or fact would answer it.

Question: {question}

Output format (one line per hop):
Hop 1: [sub-question] — needs: [type of fact/document]
Hop 2: [sub-question] — needs: [type of fact/document]
...
```

---

## Common Pitfalls

### Pitfall 1 — Interleaving Relevant and Irrelevant Documents
Placing irrelevant documents between the two gold documents triggers the "lost in between" effect. Always cluster relevant documents together.

**Wrong**: `[Gold A] [Irrel 1] [Irrel 2] [Irrel 3] [Gold B]`
**Right**: `[Gold A] [Gold B] [Irrel 1] [Irrel 2] [Irrel 3]`

### Pitfall 2 — Assuming Retrieval Order = Reasoning Order
Most retrieval systems return documents ranked by relevance score, not by reasoning-chain order. A document highly relevant to the final answer may rank first but should be placed second in the prompt.

Always run question decomposition (Template D) to determine reasoning order before constructing the prompt.

### Pitfall 3 — Using Too Many Permutations
Permutation sampling costs N API calls. More than 6 permutations rarely improves majority-vote accuracy while multiplying cost. The paper's heuristic used all permutations of the gold documents; for 2 gold docs that is only 2 permutations. Use 6 as a practical upper bound when including irrelevant docs in the permutation.

### Pitfall 4 — Dropping the First-Hop Document
If context length forces truncation, the intuitive choice (drop the document most distant from the final answer) is correct: drop the last-hop document, not the first-hop document. The first-hop document establishes the intermediate entity that the whole reasoning chain depends on.

---

## Benchmark Context

The paper uses two standard MHQA datasets:
- **MuSiQue**: 2-hop questions, requires synthesizing facts from two documents
- **HotpotQA**: 2-hop questions with a distractor document set

Reported metrics: Exact Match (EM) and F1 on held-out test sets. The permutation heuristic improvement (+5.1% absolute on Qwen 7B) was measured on MuSiQue.
