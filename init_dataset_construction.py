import requests
import csv
import os

def save_csv_locally(movie_list_3_columns):
    filename = "top_chinese_movies_data.csv"    # Initial dataset for us voting
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['English Title', 'Chinese Title', 'Overview'])
        writer.writerows(movie_list_3_columns)


API_KEY = "ENTER_YOUR_API_KEY_HERE"
base_url = "https://api.themoviedb.org/3/discover/movie"
all_movies_for_csv = []

try:
    for page in range(1, 12):  # Pages 1 to 11
        params = {
            'api_key': API_KEY,
            'with_original_language': 'zh',
            'sort_by': 'popularity.desc',
            'page': page,
            'language': 'en-US',
            'primary_release_date.gte': '2000-01-01',
            'primary_release_date.lte': '2025-12-31'
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        movies_on_page = data.get('results', [])

        if not movies_on_page:
            break

        for movie in movies_on_page:
            english_title = movie.get('title', 'N/A')
            original_title = movie.get('original_title', 'N/A')
            overview = movie.get('overview', 'No summary available.')

            all_movies_for_csv.append([english_title, original_title, overview])
        
    if all_movies_for_csv:
        save_csv_locally(all_movies_for_csv)
    else:
        print("No movie data was returned to save.")

except requests.exceptions.RequestException as e:
    print(f"API communication error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")