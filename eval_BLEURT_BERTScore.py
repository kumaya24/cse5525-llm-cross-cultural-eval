"""
@author: yingy
"""

import pandas as pd
import evaluate
import torch
import numpy as np


# Path to the CSV file (should iterate over each of the nine files)
INPUT_CSV = r"Llama_Title-only.csv"

# Output CSV with metric scores
OUTPUT_CSV = r"Llama_Title-only_metrics_scored.csv"

# Column indices
REF_COL_IDX = 0    # 1st column: reference
PRED_COL_IDX = 3   # 4th column: prediction

# BERTScore settings
BERTSCORE_LANG = "en"
BERTSCORE_MODEL = None       

# BLEURT settings
BLEURT_CHECKPOINT = "bleurt-20"

def main():
    
    df = pd.read_csv(INPUT_CSV)

    # Extract reference and prediction columns
    references = df.iloc[:, REF_COL_IDX].astype(str).fillna("").str.strip().tolist()
    predictions = df.iloc[:, PRED_COL_IDX].astype(str).fillna("").str.strip().tolist()

    print(f"Loaded {len(predictions)} samples.")

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- BLEURT ----
    bleurt = evaluate.load("bleurt", checkpoint=BLEURT_CHECKPOINT)
    bleurt_result = bleurt.compute(
        predictions=predictions,
        references=references 
    )
    bleurt_scores = bleurt_result["scores"]
    bleurt_arr = np.array(bleurt_scores)

    # ---- BERTScore ----
    bertscore = evaluate.load("bertscore")
    bert_kwargs = dict(
        predictions=predictions,
        references=references,
        lang=BERTSCORE_LANG,
        device=device 
    )
    if BERTSCORE_MODEL is not None:
        bert_kwargs["model_type"] = BERTSCORE_MODEL
    bert_result = bertscore.compute(**bert_kwargs)
    bert_p = np.array(bert_result["precision"])
    bert_r = np.array(bert_result["recall"])
    bert_f1 = np.array(bert_result["f1"])
    
    # ---- Save per-example scores ----
    df["reference_used"] = references
    df["prediction_used"] = predictions
    df["BLEURT"] = bleurt_arr
    df["BERTScore_P"] = bert_p
    df["BERTScore_R"] = bert_r
    df["BERTScore_F1"] = bert_f1
    
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    # ---- Print summary (mean ± std) ----
    print("\n=== Evaluation Summary (mean ± std) ===")
    print(f"BLEURT:        {bleurt_arr.mean():.4f} ± {bleurt_arr.std(ddof=1):.4f}")
    print(f"BERTScore P:   {bert_p.mean():.4f} ± {bert_p.std(ddof=1):.4f}")
    print(f"BERTScore R:   {bert_r.mean():.4f} ± {bert_r.std(ddof=1):.4f}")
    print(f"BERTScore F1:  {bert_f1.mean():.4f} ± {bert_f1.std(ddof=1):.4f}")

    print(f"\nSaved results to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
