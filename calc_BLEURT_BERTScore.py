"""
@author: yingy
"""

import pandas as pd
import evaluate
import torch
import numpy as np



# Input files (9 datasets)
INPUT_CSVS = [
    r"Gemini_Culture-aware.csv", r"Gemini_Title and Synopsis.csv", r"Gemini_Title-only.csv",
    r"gpt_Culture-aware.csv", r"gpt_Title_and_Synopsis.csv", r"gpt_Title-only.csv",
    r"Llama_Culture-aware.csv", r"Llama_Title_and_Synopsis.csv", r"Llama_Title-only.csv",
]

# OUTPUT: combined per-case scores
OUTPUT_CSV = r"combined_BLEURT_BERTScore_per_case.csv"

# Column indices
KEY_COL_1 = 0    # case identifier (English title)
KEY_COL_2 = 1    # case identifier (Chinese title)
REF_COL_IDX = 0  # reference (golden rule)
PRED_COL_IDX = 3 # prediction

# Metric settings
BERTSCORE_LANG = "en"
BERTSCORE_MODEL = None
BLEURT_CHECKPOINT = "bleurt-20"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bleurt = evaluate.load("bleurt", checkpoint=BLEURT_CHECKPOINT)
    bertscore = evaluate.load("bertscore")
    combined_scores = {}

    for input_csv in INPUT_CSVS:
        tag = input_csv.split("/")[-1].replace(".csv", "")
        print(f"\nProcessing: {tag}")

        df = pd.read_csv(input_csv)

        references = df.iloc[:, REF_COL_IDX].astype(str).fillna("").str.strip().tolist()
        predictions = df.iloc[:, PRED_COL_IDX].astype(str).fillna("").str.strip().tolist()

        keys = list(zip(
            df.iloc[:, KEY_COL_1].astype(str),
            df.iloc[:, KEY_COL_2].astype(str)
        ))

        # ---- BLEURT ----
        bleurt_result = bleurt.compute(
            predictions=predictions,
            references=references
        )
        bleurt_scores = np.array(bleurt_result["scores"])

        # ---- BERTScore ----
        bert_kwargs = dict(
            predictions=predictions,
            references=references,
            lang=BERTSCORE_LANG,
            device=device
        )
        if BERTSCORE_MODEL is not None:
            bert_kwargs["model_type"] = BERTSCORE_MODEL

        bert_result = bertscore.compute(**bert_kwargs)
        bert_f1 = np.array(bert_result["f1"])

        # ---- Store per-case results ----
        for k, b, f1 in zip(keys, bleurt_scores, bert_f1):
            if k not in combined_scores:
                combined_scores[k] = {
                    "Column 1": k[0],
                    "Column 2": k[1]
                }

            combined_scores[k][f"BLEURT_{tag}"] = b
            combined_scores[k][f"BERTScore_{tag}"] = f1

    # ---- output ----
    output_df = pd.DataFrame.from_dict(combined_scores, orient="index")
    output_df.reset_index(drop=True, inplace=True)

    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved combined per-case results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


