# Cross-Cultural Evaluation of LLMs: **A Mixed-Method Research of Chinese-to-English Movie Title Translation**

### Author
Jacqui Wang, Jialing Wu, Yingyu Cheng 

**Course:** CSE 5525 - Foundations of Speech and Language Processing

> **Note:** This project served as term project for CSE 5525. 
---
### Project Overview

This repository contains the code for creating dataset, and evaluating how Large Language Models (LLMs) handle the cultural adaptation required in translating Chinese film titles into English.

Unlike standard machine translation tasks, film titles require balancing semantic fidelity with cultural relevance and commercial appeal. This project evaluates three state-of-the-art LLMs using a mixed-method approach, combining embedding-based metrics with a qualitative analysis of cultural adaptation strategies.

---
### Models

We evaluated the following models on their ability to translate culturally specific items (CSIs):

  * **GPT-5-chat**
  * **Gemini-2.5-flash-preview-09-2025** 
  * **Llama\_3.1\_8b\_instant** 
---
### Prompting Strategies

We tested three distinct prompting conditions to measure the impact of context:

1.  **Title-only:** Zero-shot translation of the Chinese title.
2.  **Title + Synopsis:** Providing the plot summary to test contextual understanding.
3.  **Culture-aware:** Explicit instructions to preserve or transform cultural elements and explain the rationale.
---
###  Dataset

  * **Source:** Data was collected via the TMDB API and manually verified.
  * **Scope:** Chinese films released between 2000 and 2025.
  * **Filtering:** The final dataset comprises **81 film titles** that contain specific cultural elements or require cultural adaptation.
  * **Fields:** Chinese Title, Official English Title, Chinese Synopsis.
---
###  Evaluation Metrics

This project utilizes a custom evaluation framework:

**1. Quantitative Metrics**

  * **Cosine Similarity:** Uses `all-MiniLM-L6-v2` to measure broad semantic agreement with official English titles.
  * **CSI-Match:** A fuzzy-matching metric (adapted from Yao et al., 2024) designed to evaluate the preservation of Culturally Specific Items (CSIs).
  * **TODO**

**2. Qualitative Analysis**
Translations were manually coded into four cultural adaptation strategies:

  * **Preservation:** Retaining specific cultural elements (e.g., transliteration).
  * **Transformation:** Reinterpreting cultural imagery for the target audience.
  * **Omission:** Removing cultural features for simplicity.
  * **Mistranslation:** Inaccurate rendering of meaning.
---
### Key Findings

  * **Context is Key:** The **GPT + Title + Synopsis** configuration achieved the highest performance ($\mu_{CSI}=0.80$), suggesting that narrative grounding is essential for cultural interpretation[cite: 133, 134].
  * **Low-Resource Stability:** While GPT excelled with context, **Gemini** performed best in the **Title-only** (low-context) condition[cite: 135].
  * **Adaptation Strategies:** The best-performing models favored **Preservation** and **Transformation** strategies over Omission[cite: 187].
---

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cse5525-llm-cross-cultural-eval.git
cd cse5525-llm-cross-cultural-eval

# Install dependencies
pip install -r requirements.txt
```
###  Usage

*TODO*

```bash
# Install dependencies
pip install -r requirements.txt
```
