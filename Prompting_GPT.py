import csv
import json
import time
import requests
import os
from typing import List, Dict

INPUT_FILE = 'Final dataset(Sheet1).csv'    # Filtered dataset of movie titles
AZURE_ENDPOINT = 'https://[YOUR-AZURE-RESOURCE-NAME].openai.azure.com/'
AZURE_DEPLOYMENT_NAME = 'gpt-5-chat' 
API_KEY = 'YOUR_SECRET_AZURE_API_KEY_HERE' 
API_VERSION = '2024-02-01'
MODEL = AZURE_DEPLOYMENT_NAME
AZURE_API_URL = f'{AZURE_ENDPOINT}openai/deployments/{MODEL}/chat/completions?api-version={API_VERSION}'

OUTPUT_DIR = "outputs"

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
    if "YOUR_SECRET" in API_KEY or "[YOUR-" in AZURE_ENDPOINT:
        return "ERROR: Please update your Azure Configuration in the script!"

    payload = {
        "messages": [
        {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,
    }

    headers = {
        'Content-Type': 'application/json',
        'api-key': API_KEY 
    }
  
    for attempt in range(max_retries):
        try:
            response = requests.post(AZURE_API_URL, headers=headers, data=json.dumps(payload))
        
            # Success
            if response.status_code == 200:
                result = response.json()
                if result.get('choices') and result['choices'][0].get('message'):
                    return result['choices'][0]['message']['content'].strip()
            
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"Rate Limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "ERROR: Failed after multiple retries."
                
            elif response.status_code == 401:
                return "ERROR: Authentication Failed. Check API Key."
                
            else:
                print(f"HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
        
    return "ERROR: Failed after multiple retries."

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
        
    return all_results

def write_csv(fn: str, fieldnames: List[str], data: List[Dict[str, str]]) -> str:
    full_path = os.path.join(OUTPUT_DIR, fn)
    try:
        with open(full_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return full_path
    except Exception as e:
        print(f"ERROR writing to CSV {full_path}: {e}")
        return ""


all_data_rows = []
with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames:
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
    all_data_rows = list(reader)
    
output_fieldnames = ['English Title', 'Chinese Title', 'Overview', 'Prompt_Response']
results = process_data(all_data_rows, PROMPTS) 

for p_num, results in results.items():
    fn = f"azure_{AZURE_DEPLOYMENT_NAME.replace('-', '_')}_prompt{p_num}_{PROMPTS[p_num]['title']}.csv"
    write_csv(fn, output_fieldnames, results)