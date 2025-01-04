import requests
import time
from typing import List
import json
import os
from pathlib import Path

def is_id_already_downloaded(number: int, base_folder: str = "crawled_data") -> bool:
    """
    Check if the ID has already been downloaded by checking folder existence
    and validating the data file exists and is not empty.
    """
    folder_path = Path(base_folder) / str(number)
    data_file = folder_path / "data.json"

    # Check if both folder and file exist
    if folder_path.exists() and data_file.exists():
        # Optional: Verify file is not empty and contains valid JSON
        try:
            if data_file.stat().st_size > 0:
                with open(data_file, 'r', encoding='utf-8') as f:
                    json.load(f)  # Try to load JSON to verify it's valid
                return True
        except (json.JSONDecodeError, OSError):
            print(f"Warning: Corrupt or empty file found for ID {number}")
            return False
    return False

def save_response(number: int, data: dict, base_folder: str = "crawled_data") -> None:
    """
    Saves the response data in a dedicated folder with the ID as folder name.
    """
    folder_path = Path(base_folder) / str(number)
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / "data.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Saved data to {file_path}")

def crawl_nearby_urls(base_url: str, start_number: int, range_to_check: int = 100,
                     delay: float = 0.5, base_folder: str = "crawled_data") -> List[int]:
    """
    Crawls URLs with nearby numbers and saves each response in its own folder.
    Skips already downloaded IDs.
    """
    successful_ids = []
    checked_numbers = set()

    # Load existing successful IDs from index file if it exists
    index_path = Path(base_folder) / "index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                existing_data = json.load(f)
                successful_ids = existing_data.get("crawled_ids", [])
                print(f"Loaded {len(successful_ids)} existing IDs from index")
        except json.JSONDecodeError:
            print("Warning: Could not load existing index file")

    def try_url(number: int) -> None:
        if number in checked_numbers:
            return

        checked_numbers.add(number)

        # Skip if already downloaded
        if is_id_already_downloaded(number, base_folder):
            print(f"Skipping {number} - already downloaded")
            if number not in successful_ids:
                successful_ids.append(number)
            return

        url = base_url.format(number)

        try:
            response = requests.get(url)
            if response.status_code == 200:

                # print(f"Success: {url}")
                data = response.json()
                if data["name"]=="lux_ai_s3":
                    save_response(number, data, base_folder)
                    successful_ids.append(number)
                # "name": "lux_ai_s3","
                else:
                    print(f"Different game {number} : {data['name']}")

            else:
                print(f"Failed ({response.status_code}): {url}")

        except Exception as e:
            print(f"Error accessing {url}: {str(e)}")

        time.sleep(delay)

    # Create base folder if it doesn't exist
    Path(base_folder).mkdir(parents=True, exist_ok=True)

    # Check the start number first
    try_url(start_number)

    # Check numbers in both directions with increasing distance
    for offset in range(1, range_to_check):
        # Check number above
        try_url(start_number + offset)

        # Check number below
        try_url(start_number - offset)

        # Optional: Stop if we haven't found anything new in a while
        if offset > 20 and len(successful_ids) == 0:
            print("No successful responses found in last 20 attempts, stopping...")
            break

    return sorted(list(set(successful_ids)))  # Remove duplicates and sort

def create_index_file(successful_ids: List[int], base_folder: str = "crawled_data") -> None:
    """
    Creates an index file listing all successfully crawled IDs.
    """
    index_path = Path(base_folder) / "index.json"
    index_data = {
        "crawled_ids": sorted(successful_ids),
        "total_count": len(successful_ids),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)

    print(f"Created index file at {index_path}")

def main():
    BASE_URL = "https://www.kaggleusercontent.com/episodes/{}.json"
    START_NUMBER = 61369571
    BASE_FOLDER = "crawled_data"

    successful_ids = crawl_nearby_urls(
        base_url=BASE_URL,
        start_number=START_NUMBER,
        range_to_check=50,
        delay=0.5,
        base_folder=BASE_FOLDER
    )

    create_index_file(successful_ids, BASE_FOLDER)

    print(f"\nCrawling completed:")
    print(f"- Total successful IDs: {len(successful_ids)}")
    print(f"- Data saved in: {BASE_FOLDER}")

if __name__ == "__main__":
    main()
