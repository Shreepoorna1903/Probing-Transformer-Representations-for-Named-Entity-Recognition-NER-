# Layer-wise Probing of BERT Representations for Named Entity Recognition

This project investigates **where linguistic and semantic information emerges inside BERT** by performing **layer-wise probing** on a frozen `bert-base-cased` model using the **CoNLL-2003 Named Entity Recognition (NER)** task.

We train lightweight linear probes on hidden states from each transformer layer to predict NER labels, achieving a peak **token-level accuracy of 97.35%**, with performance improving across layers.

---

## Overview

**Probing** is an analysis technique used to study what information is encoded inside pretrained language models without fine-tuning them.

In this project:
- BERT weights are **fully frozen**
- Only **linear classifiers (Logistic Regression)** are trained
- Probes are applied **independently to each of BERT’s 13 layers**
- The goal is to understand *how representations evolve across layers*

---

## Dataset

- **CoNLL-2003 (English)**
- Task: Named Entity Recognition (NER)
- Entity tags (BIO format):
  - PER (Person)
  - ORG (Organization)
  - LOC (Location)
  - MISC (Miscellaneous)

We work with a **subset of 1,000 sentences** for computational efficiency.

---

## Methodology

### 1. Tokenization & Label Alignment
- Tokenization via `bert-base-cased`
- Word-level NER labels aligned to **subword tokens**
- Special tokens (`[CLS]`, `[SEP]`) and continuation subwords assigned label `-100` (ignored)

---

### 2. Extracting Layer-wise Representations
- BERT outputs **13 hidden layers**:
  - Layer 0 → word embeddings
  - Layers 1–12 → transformer layers
- For each token, we extract a **768-dimensional vector from every layer**

Final representation shape:

