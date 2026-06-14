# WordSalad: Detecting Question Evasion in Political QA Pairs

## Project Overview

This project systematically detects and analyzes political question-answer (QA) pairs, identifying **3 clarity classes** and **9 fine-grained evasion types**:

| Level | Categories |
|-------|-------------|
| **Clarity (3 classes)** | Clear reply, Ambivalent reply, Clear non-reply |
| **Evasion (9 types)** | *Clear reply*: Explicit<br>*Ambivalent reply*: Implicit, Dodging, General, Deflection, Partial<br>*Clear non-reply*: Declining, Ignorance, Clarification |

We compare multiple model architectures — from statistical baselines to LLMs — and perform reasoning chain analysis to identify failure patterns.

**Dataset**: [QEvasion on HuggingFace](https://huggingface.co/datasets/QEvasion)

---

## Project Goals

This study expands existing methods for predicting and analyzing question-answer clarity into the political domain, centered on three research questions:

1. How do LLMs perform on political QA tasks —  
   (a) Evasion-based classification (9 labels)  
   (b) Clarity-based classification (3 labels)  
   compared to compute-light approaches (TF-IDF + Logistic Regression) and sequence classification encoders (ModernBERT)?

2. How do inference-style prompting strategies — including few-shot and role-playing methods — affect classification quality?

3. How does detection difficulty vary across evasion types across different models, and what common error patterns emerge?

---

## Models Evaluated

| Paradigm | Models / Methods |
|----------|------------------|
| **Statistical** | TF-IDF + Logistic Regression |
| **Encoder** | ModernBERT |
| **Decoder (LLM)** | DeepSeek-R1-Distill (1.5B, 7B, 8B, 14B, 32B, 70B) |
| **Prompting Strategies** | Zero-shot, Few-shot, Role-playing |
| **Baselines** | Random / Majority Class, Statistical model |

---

## Project Structure

```bash
WordSalad/
├── Deepseek_master.py               # DeepSeek LLM inference (zero-shot & few-shot)
├── ModernBert_master.py             # ModernBERT model inference
├── bi-encoder_master.ipynb          # Bi-Encoder (all-mpnet-base-v2, multi-qa-mpnet-base-cos-v1) for noise filtering
├── dummybaseline_master.ipynb       # Random / majority class baseline
├── tf_idf_master.py                 # TF-IDF + traditional classifier
├── multiagrnt_master.py             # Role-playing framework
├── final_figure_making.ipynb        # Visualization & result aggregation
├── test_set_me.csv                  # Test set (QA pairs + labels)
└── README.md

## Citation

If you use this code or the QEvasion dataset, please cite the original paper:

```bibtex
@inproceedings{thomas-etal-2024-never,
    title = "``{I} Never Said That'': A dataset, taxonomy and baselines on response clarity classification",
    author = "Thomas, Konstantinos and Filandrianos, Giorgos and Lymperaiou, Maria and Zerva, Chrysoula and Stamou, Giorgos",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.300/",
    doi = "10.18653/v1/2024.findings-emnlp.300",
    pages = "5204--5233"
}
