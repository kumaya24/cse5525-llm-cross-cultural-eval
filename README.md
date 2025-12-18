# Cross-Cultural Evaluation of LLMs  
## A Mixed-Method Study of Chinese-to-English Movie Title Translation

**Authors:** Jacqui Wang · Jialing Wu · Yingyu Cheng (The Ohio State University)  
**Course context:** CSE 5525 – Foundations of Speech and Language Processing  
> This project originated as a course term project and was expanded into a full mixed-method research study.

---

## Project Overview

This repository provides the dataset construction pipeline and evaluation code for a cross-cultural study of how large language models (LLMs) translate **Chinese movie titles into English**.

Movie title translation is a culturally dense, highly constrained task. Unlike standard machine translation, titles must balance **semantic fidelity**, **cultural relevance**, and **audience accessibility / commercial appeal**. Many titles contain culturally specific items (CSIs), idioms, historical allusions, genre conventions (e.g., Wuxia), or condensed metaphors that have no direct equivalents in English.

We evaluate LLM translation behavior using a **mixed-method framework**:
- **Quantitative evaluation** with semantic and culture-sensitive metrics  
- **Qualitative analysis** of cultural adaptation strategies (Preservation/Transformation/Omission/Mistranslation)

---

## Related Context (Why This Matters)

Recent work increasingly treats **cultural adaptation** as a distinct challenge for LLM-based translation. Prior studies show that:
- **Prompting** and **culture-aware resources** can improve CSI translation,
- **Retrieval-augmented approaches** (e.g., multilingual knowledge graphs) help with culturally nuanced names,
- LLMs may default to **English-centric cultural knowledge**, motivating explicit benchmarks and evaluation tools.

However, **film title translation** remains underexplored in LLM research, despite its uniquely high cultural density and commercial constraints. This repository supports systematic evaluation in this specific domain.

---
## Installation
Clone and install requirements:
   ```bash
   git clone https://github.com/kumaya24/cse5525-llm-cross-cultural-eval
   pip install -r requirements.txt
```
---
## Usage
> All output are under dataset/ directory
1. Construct the dataset of reference
   ```bash
   python init_dataset_construction.py
   ```
2. Construct the dataset of hypothesis
   ```bash
   python Prompting_GPT.py
   python Prompting_Gemini.py
   python Prompting_Llama.py
   ```
3. Calculate scores by all metrics
   ```bash
   python calc_BLEURT_BERTSore.py
   python calc_cosine_similarity.py
   python calc_csi_match_psr.py
   ```
4. Evalutae scores of all metrics
   ```bash
   python calc_p_value.py
   python eval_cs.py
   python eval_csi.py
   python eval_BLEURT_BERTScore.py
   ```
## Models Evaluated

We evaluate three models under controlled prompting conditions:

- **GPT-5-chat** (referred to as *GPT*)
- **gemini-2.5-flash-preview-09-2025** (referred to as *Gemini*)
- **llama_3.1_8b_instant** (referred to as *Llama*)

**Important control:** Models are instructed **not** to look up or use official English titles.

---

## Prompting Strategies

We design three prompt conditions to test the impact of context and explicit cultural guidance:

1. **Title-only**  
   Translate using only the Chinese title (no context).

2. **Title + Synopsis**  
   Translate using the Chinese title plus a brief Chinese synopsis (narrative grounding).

3. **Culture-aware**  
   Translate with explicit instructions to consider meaning, tone, and cultural context, and to preserve/transform cultural elements (with a brief rationale).

All prompts are **standardized in Chinese** to reduce prompt-language bias.

---

## Dataset

- **Source:** TMDB public API (publicly accessible), followed by manual verification  
- **Time span:** Chinese films released between **2000–2025**  
- **Filtering (two rounds):**
  1. Inclusion/exclusion rules (official Chinese + official English titles; year constraints; popularity-based retrieval; remove re-releases/special editions; remove non-Chinese originals).
  2. Manual cultural annotation by all authors; titles are retained if **≥ 2 annotators** judge the Chinese title as containing cultural elements.

- **Final dataset:** **81 film titles**  
- **Fields:**  
  - Chinese Title  
  - Official English Title (reference only)  
  - Chinese Synopsis  

---

## Evaluation Framework

Official English titles are treated as the **reference standard** for evaluation (with caveats; see Limitations).

### Quantitative Metrics

We use four quantitative metrics to capture both general semantic alignment and CSI-sensitive matching:

- **BLEURT** (Sellam et al., 2020): model-based semantic evaluation  
- **BERTScore (F1)** (Zhang et al., 2020): contextual embedding similarity  
- **Cosine Similarity (CS)**: sentence-embedding similarity using `all-MiniLM-L6-v2`  
- **CSI-Match** (adapted from Yao et al., 2024): fuzzy matching for culture-specific items  
  - computes maximum partial similarity ratio using normalized Levenshtein distance  
  - targets culturally bound terms (names, idioms, culturally marked references)

### Qualitative Coding (Cultural Adaptation Strategies)

For the best-performing model–prompt configuration, translations are manually coded into four strategies:

1. **Preservation**: retain culturally specific elements (e.g., transliteration, named entities)  
2. **Transformation**: adapt cultural imagery into an accessible English expression while preserving intent  
3. **Omission**: remove/downplay cultural elements for fluency/simplicity  
4. **Mistranslation**: inaccurate interpretation of cultural meaning

We report the distribution across strategies to make cultural adaptation behavior interpretable.

---

## Key Results (From the Paper)

### Quantitative Findings

- We evaluate **3 models × 3 prompt conditions × 81 titles = 729 outputs**.
- Two-way ANOVA (Model × Prompt Condition) shows:
  - significant model differences for **CSI-Match** and **CS**
  - significant prompt-condition effects
  - **significant interaction**: performance depends strongly on which prompt condition is used.

**Best overall configuration:**  
- **GPT + Title + Synopsis** achieves the strongest overall performance across metrics, including:
  - BLEURT mean **-0.05**
  - BERTScore mean **0.93**
  - CSI-Match mean **0.80**  
  - CS mean **0.75**

**Low-context setting:**  
- Under **Title-only**, **Gemini** performs best among models, indicating stronger stability when no synopsis is available.

### Qualitative Findings (Best Configuration: GPT + Title + Synopsis)

Strategy counts (N = 81):
- **Preservation:** 46  
- **Transformation:** 33  
- **Omission:** 11  
- **Mistranslation:** 2  

This pattern suggests GPT tends to preserve explicit CSIs and transform culturally heavy imagery when possible, but may still fail on deeply embedded socio-historical allusions.

---

## Prompt Templates (Appendix A)

- Models are instructed not to reference official English titles.
- Prompts are standardized in Chinese.
- Official English titles are used **only for evaluation**.

---

We provide the three prompt templates (Chinese input + English gloss for readability).
Note: The Chinese prompt is the actual model input; English is for readers only.

1) Title-only

Chinese (LLM input):

请直接根据所给的中文电影片名翻译为英文。不要查找或使用该电影的任何外部信息，包括简介、评论或官方英文译名。只输出一个英文片名，不要添加解释、说明或其他内容。

English (for readers):

Please translate the given Chinese movie title into English. Do not look up or use any external information about the film, including plot summaries, reviews, or official English titles. Output only one English title. Do not add any explanation, description, or additional text.

2) Title + Synopsis

Chinese (LLM input):

请根据提供的剧情简介将以下中文电影标题翻译成英文。请不要查找或使用关于该电影的任何外部信息，包括官方网站、新闻文章或观众评论。仅输出一个英文标题。不要添加任何解释、描述或额外文本。

English (for readers):

Please translate the following Chinese movie title into English based on the provided plot summary. Do not look up or use any external information about the film, including official websites, news articles, or viewer comments. Output only one English title. Do not add any explanation, description, or additional text.

3) Culture-aware

Chinese (LLM input):

请将以下中文电影片名翻译为英文。不要查找或使用该电影的官方英文译名。请避免逐字直译，应结合语义、语气与文化背景，尽量保留或恰当地转化原片名中蕴含的中国文化元素，使译名在英文语境中既自然流畅，又能体现原片名的文化意涵。译完后，请简要说明你的翻译理由（不超过两句话）。

English (for readers):

Please translate the following Chinese movie title into English. Do not look up or use the film’s official English title. Avoid literal translation; instead, consider the meaning, tone, and cultural context. Try to preserve or appropriately transform the cultural elements embedded in the original Chinese title so that the English translation reads naturally while conveying the cultural connotations of the source title. After translating, briefly explain your translation rationale in no more than two sentences.
