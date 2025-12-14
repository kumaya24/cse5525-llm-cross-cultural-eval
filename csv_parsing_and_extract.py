import csv
from typing import List, Dict, Any
from io import StringIO
import re

INPUT_FILE = 'modelName_Culture-aware.csv'  # NEED TO REVISE FOR DESIRED FILES
OUTPUT_FILE = 'modelName_Culture-aware_Parsed.csv'  # NEED TO REVISE FOR DESIRED FILES

def parse_response(response_text: str) -> Dict[str, str]:
    if not response_text or "ERROR" in response_text:
        return {"Extracted_English_Title": "N/A", "Translation_Reason": response_text}
    
    title_match = re.search(r"英文片名[:：]\s*(.*?)(?:\n\n|\n|理由[:：]|$)", response_text, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r"(?:翻译理由[:：]|理由[:：])\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    
    extracted_title = ""
    if title_match:
        extracted_title = title_match.group(1).strip().replace('\n', ' ')
        
    extracted_reason = ""
    if reason_match:
        extracted_reason = reason_match.group(1).strip().replace('\n', ' ')

    return {
        "Extracted_English_Title": extracted_title,
        "Translation_Reason": extracted_reason
    }


# Read
with open(INPUT_FILE, 'r', newline='', encoding='utf-8-sig') as infile:
    reader = csv.DictReader(infile)
    
    og_fieldnames = reader.fieldnames
    new_fieldnames = og_fieldnames + ["Extracted_English_Title", "Translation_Reason"]
    
    processed_rows = []
    
    for row in reader:
        raw_response = row.get('Prompt_Response', '')
        parsed_data = parse_response(raw_response)
        
        new_row = row.copy()
        new_row.update(parsed_data)
        
        processed_rows.append(new_row)

# Write
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(processed_rows)
    
