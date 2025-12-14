import csv
import os
from sentence_transformers import SentenceTransformer, util
import collections
import statistics 
from math import fabs 

FILE_NAMES = [
    'datasets/Gemini_Culture-aware.csv', 'datasets/Gemini_Title_and_Synopsis.csv', 'datasets/Gemini_Title-only.csv', 
    'datasets/gpt_Culture-aware.csv', 'datasets/gpt_Title_and_Synopsis.csv',
    'datasets/gpt_Title-only.csv', 'datasets/Llama_Culture-aware.csv', 
    'datasets/Llama_Title_and_Synopsis.csv', 'datasets/Llama_Title-only.csv'
]
OUTPUT_FILE = "datasets/combined_similarity_scores.csv"

COLUMN_INDEX_A = 0
COLUMN_INDEX_B = 3
KEY_COLUMN_INDICES = [0, 1]

model = SentenceTransformer('all-MiniLM-L6-v2')

agg_scores = collections.defaultdict(lambda: {name: '' for name in FILE_NAMES})
total_processed_rows = 0
successful_files = 0

for fn in FILE_NAMES:
    with open(fn, 'r', newline='', encoding='utf-8-sig') as f_in:
        reader = csv.reader(f_in)
        
        header = next(reader)
        rows = list(reader)
        current_rows = 0
        
        for row in rows:
            key = (row[KEY_COLUMN_INDICES[0]], row[KEY_COLUMN_INDICES[1]])
            text_a = row[COLUMN_INDEX_A] # Column 1
            text_b = row[COLUMN_INDEX_B] # Column 4

            embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            agg_scores[key][fn] = f"{score:.4f}"
            current_rows += 1
                
        if current_rows > 0:
            total_processed_rows += current_rows
            successful_files += 1
        else:
            print(f"WARNING: File '{fn}' processed, but yielded 0 valid rows for similarity calculation.")

output_header = [
    'Column 1', 
    'Column 2',
] + [name.split('/')[1] for name in FILE_NAMES]

output_rows = []
for key, scores in agg_scores.items():
    row = list(key)
    for fn in FILE_NAMES:
        row.append(scores.get(fn, '')) 

    output_rows.append(row)

if output_rows:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_header)
        writer.writerows(output_rows)
else:
        print("\n No output file written.")