import csv
import json
import time
import requests
from typing import List, Dict, Any
import os 
from pathlib import Path

INPUT_FILE = 'Final dataset(Sheet1).csv'    # Filtered dataset of movie titles
API_KEY = 'YOUR_SECRET_GROQ_API_KEY_HERE' 
GROQ_MODEL = 'llama-3.1-8b-instant' 
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
MODEL = GROQ_MODEL 

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

def call_gpt_api(user_query: str, max_retries: int = 10) -> str:
    payload = {
        "model": MODEL, 
        "messages": [
            {"role": "user", "content": user_query}
        ],
     "temperature": 0.0,
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}' 
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 
            
            result = response.json()
            
            if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"API returned no text content: {json.dumps(result, indent=2)}")
                return "ERROR: API returned empty or unexpected response."

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue 
                else:
                    print(f"Failed after {max_retries} attempts.")
                    return "ERROR: Failed after multiple retries due to rate limit (429)."
                    
            elif response.status_code == 401:
                print(f"Authentication Error. Check your API Key.")
                return "ERROR: Authentication Failed."
            elif response.status_code == 400:
                print(f"Bad Request. Check Model Name and Payload. Response: {response.text}")
                return "ERROR: Bad Request (400)."
            else:
                print(f"HTTP Error on attempt {attempt + 1}: {e}")
                return f"ERROR: HTTP Status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error on attempt {attempt + 1}: {e}")
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
            
        print(f"Row {i+1}/{len(data_rows)}: Processing '{chinese_title}'...")

        template_data = {
            "chinese_title": chinese_title,
            "overview": overview
        }

        for p_num, p_config in prompts.items():
            
            if "overview" in p_config["data_columns"] and not overview:
                response_text = "ERROR: Skipping query for missing Overview data."
                print(f"  > Prompt {p_num} ({p_config['title']}): Skipping due to missing Overview.")
            else:
                user_query = p_config['user_query_template'].format(**template_data)
                response_text = call_gpt_api(user_query) 
            
            result_row = {
                'English Title': row.get('English Title', ''),
                'Chinese Title': chinese_title,
                'Overview': overview, 
                'Prompt_Response': response_text
            }
            
            all_results[p_num].append(result_row)
            if "ERROR" not in response_text:
                print(f"  > Prompt {p_num} ({p_config['title']}): Response received.")
            
    return all_results

def write_csv(fn: str, fieldnames: List[str], data: List[Dict[str, str]]) -> str:
    try:
        with open(fn, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return fn
    except Exception as e:
        print(f"ERROR writing to CSV {fn}: {e}")
        return ""

all_data_rows = []
input_path = Path(INPUT_FILE)


with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    if reader.fieldnames:
        reader.fieldnames = [name.strip().replace('\ufeff', '') for name in reader.fieldnames]
        
        all_data_rows = list(reader)

output_fieldnames = ['English Title', 'Chinese Title', 'Overview', 'Prompt_Response']
results = process_data(all_data_rows, PROMPTS) 
for p_num, results in results.items():
    fn = f"groq_{MODEL.replace('-', '_')}_prompt{p_num}_{PROMPTS[p_num]['title']}.csv"
    write_csv(fn, output_fieldnames, results)
