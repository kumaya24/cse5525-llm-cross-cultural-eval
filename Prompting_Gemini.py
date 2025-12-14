import csv
import json
import time
import requests
from typing import List, Dict, Any
from io import StringIO 
import os 
import sys 

INPUT_FILE = 'Final dataset(Sheet1).csv'    # Filtered dataset of movie titles
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_KEY = 'YOUR_SECRET_AZURE_API_KEY_HERE' 
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

MODEL = GEMINI_MODEL
API_KEY = GEMINI_API_KEY

PROMPTS = {
    1: {
        "title": "Title-only",
        "data_columns": ["chinese_title"], 
        "user_query_template": (
            "请直接根据所给的中文电影片名翻译为英文。不要查找或使用该电影的任何外部信息，包括简介、评论或官方英文译名。只输出一个英文片名，不要添加解释、说明或其他内容。"
            "中文片名：{chinese_title}"
        )
    },
    2: {
        "title": "Title_and_Synopsis",
        "data_columns": ["chinese_title", "overview"], 
        "user_query_template": (
            "请根据提供的剧情简介将以下中文电影标题翻译成英文。请不要查找或使用关于该电影的任何外部信息，包括官方网站、新闻文章或观众评论。仅输出一个英文标题。不要添加任何解释、描述或额外文本。"
            "中文片名：{chinese_title}\n"
            "概述：{overview}"
        )
    },
    3: {
        "title": "Culture-aware",
        "data_columns": ["chinese_title", "overview"], 
        "user_query_template": (
            "请将以下中文电影片名翻译为英文。不要查找或使用该电影的官方英文译名。请避免逐字直译，应结合语义、语气与文化背景，尽量保留或恰当地转化原片名中蕴含的中国文化元素，使译名在英文语境中既自然流畅，又能体现原片名的文化意涵。译完后，请简要说明你的翻译理由（不超过两句话）。"
            "中文片名：{chinese_title}\n"
            "概述：{overview}"
        )
    }
}

def call_gemini_api(user_query: str, max_retries: int = 10) -> str: 
    api_url = GEMINI_API_URL_TEMPLATE.format(model=MODEL, key=API_KEY)
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
    }

    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 
            
            result = response.json()
            
            if result.get('candidates') and result['candidates'][0]['content']['parts'][0].get('text'):
                return result['candidates'][0]['content']['parts'][0].get('text').strip()
            else:
                if response.status_code == 429: 
                    raise requests.exceptions.HTTPError(response.text, response=response)
                return "ERROR: API returned empty or unexpected response (non-retriable)."

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 or (response.status_code == 400 and "quota" in response.text.lower()):
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt) 
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue 
                else:
                    print(f"Failed after {max_retries} attempts.")
                    return "Failed after multiple retries due to rate limit."
            elif response.status_code == 400 or response.status_code == 403:
                print(f"Check your API Key or Quota settings. Response: {response.text}")
                return f"ERROR: Authentication/Quota Failed ({response.status_code})."
            else:
                print(f"HTTP Error on attempt {attempt + 1}: {e}. Status: {response.status_code}")
                return f"ERROR: HTTP Status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error (Connection/Timeout) on attempt {attempt + 1}: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
            
    return "ERROR: Failed to get response after multiple retries."

def process_data(data_rows: List[Dict[str, str]], prompts: Dict[int, Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    all_results = {p_num: [] for p_num in prompts}
    
    for i, row in enumerate(data_rows):
        chinese_title = row.get('Chinese Title', '').strip()
        overview = row.get('Overview', '').strip()

        if not chinese_title:
            continue
            
        template_data = {
            "chinese_title": chinese_title,
            "overview": overview
        }

        for p_num, p_config in prompts.items():
            
            if "overview" in p_config["data_columns"] and not overview:
                response_text = "ERROR: Skipping query for missing Overview data."
                print(f"Prompt {p_num} ({p_config['title']}): Skipping due to missing Overview.")
            else:
                user_query = p_config['user_query_template'].format(**template_data)
                response_text = call_gemini_api(user_query) 
            
            result_row = {
                'English Title': row.get('English Title', ''),
                'Chinese Title': chinese_title,
                'Overview': overview, 
                'Prompt_Response': response_text
            }
            
            all_results[p_num].append(result_row)
            if "ERROR" not in response_text:
                print(f"Prompt {p_num} ({p_config['title']}): Response received.")
            
    return all_results

def write_csv(fn: str, fieldnames: List[str], data: List[Dict[str, str]]) -> str:
    with open(fn, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return fn


all_data_rows = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    if reader.fieldnames:
        reader.fieldnames = [name.strip().replace('\ufeff', '') for name in reader.fieldnames]
        
    all_data_rows = list(reader)

output_fieldnames = ['English Title', 'Chinese Title', 'Overview', 'Prompt_Response']
results = process_data(all_data_rows, PROMPTS) 

for p_num, results in results.items():
    fn = f"gemini_{MODEL.replace('-', '_')}_prompt{p_num}_{PROMPTS[p_num]['title']}.csv"
    write_csv(fn, output_fieldnames, results)