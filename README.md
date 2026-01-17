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
X ∈ ℝ[N_tokens, 13, 768]
y ∈ ℝ[N_tokens]

---

### 3. NER Probing Experiment
For each BERT layer:
- Train a **Logistic Regression** classifier
- Input: token embeddings from that layer
- Output: NER label
- Evaluation: **10-fold cross-validation**
- Metric: **token-level accuracy**

---

### 4. Baseline Comparison
- Baseline: always predict the majority class (`O`)
- Used to show that probe performance is not due to class imbalance

---

### 5. Capitalization Probing (Auxiliary)
- Probes whether a token is **capitalized**
- Shows that **surface features are captured early**, while semantic features emerge later
- Highlights representational differences across layers

---

## Key Results

- Token-level NER accuracy increases steadily across layers
- **Best performance occurs in upper transformer layers**
- Peak accuracy: **97.35%**
- Early layers encode surface features (e.g., capitalization)
- Deeper layers encode **semantic entity information**

These findings align with known representational hierarchies in transformer models.

---

## How to Run

1. Open `bert_layerwise_probing_ner.ipynb` in **Google Colab**
2. Set runtime to **GPU or CPU** (GPU not strictly required)
3. Run cells top-to-bottom:
   - Dataset loading
   - Tokenization + alignment
   - Representation extraction
   - Layer-wise probing
   - Baseline and capitalization experiments

All dependencies are installed inside the notebook.

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers & Datasets
- scikit-learn
- NumPy

---

## Notes

- This is an **analysis / interpretability study**, not a fine-tuned NER system
- Models are intentionally simple to isolate representational content
- Results depend on dataset subset size and cross-validation randomness

---

## Reference

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition

