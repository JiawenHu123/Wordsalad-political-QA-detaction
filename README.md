WordSalad: Detecting Question Evasion in Political QA Pairs
Project Overview
This project systematically detects and analyzes political question-answer (QA) pairs, identifying 3 clarity classes and 9 fine-grained evasion types:
LevelCategoriesClarity (3 classes)Clear reply, Ambivalent reply, Clear non-replyEvasion (9 fine-grained types)Clear reply: Explicit; Ambivalent reply: Implicit, Dodging, General, Deflection, Partial; Clear non-reply: Declining, Ignorance, Clarification
We compare multiple model architectures—from statistical baselines to LLMs—and perform reasoning chain analysis to identify failure patterns.
Dataset: QEvasion on HuggingFace

Project Goals
This study expands existing methods for predicting and analyzing question-answer clarity into the political domain, centered on three research questions:
How do LLMs perform on political QA tasks—
(a) Evasion-based classification (9 evasion labels)
(b) Clarity-based classification (3 clarity labels)
compared to compute-light approaches (TF-IDF + Logistic Regression) and sequence classification encoders (ModernBERT)?
How do inference-style prompting strategies—including few-shot and role-playing methods—affect classification quality?
How does detection difficulty vary across evasion types across different models, and what common error patterns emerge?

Models Evaluated
ParadigmModels / MethodsStatisticalTF-IDF + Logistic RegressionEncoderModernBERTDecoder (LLM)DeepSeek-R1-Distill (1.5B, 7B, 8B, 14B, 32B, 70B)Prompting StrategiesZero-shot, Few-shot, Role-playingBaselinesRandom / Majority Class, Statistical model

Project Structure
WordSalad/
├── Deepseek_master.py # DeepSeek LLM inference (zero-shot & few-shot)
├── ModernBert_master.py # ModernBERT model inference
├── bi-encoder_master.ipynb # Bi-Encoder (all-mpnet-base-v2, multi-qa-mpnet-base-cos-v1) for noise filtering
├── dummybaseline_master.ipynb # Random / majority class baseline
├── tf_idf_master.py # TF-IDF + traditional classifier
├── multiagrnt_master.py # Role-playing framework
├── final_figure_making.ipynb # Visualization & result aggregation
├── test_set_me.csv # Test set (QA pairs + labels)
└── README.md
Note: All model outputs and evaluation results are generated via final_figure_making.ipynb.
