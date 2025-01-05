import requests
import time
from typing import List
import json
import os
from pathlib import Path
import random

def is_id_already_downloaded(number: int, base_folder: str = "crawled_data") -> bool:
    folder_path = Path(base_folder) / str(number)
    data_file = folder_path / "data.json"

    if folder_path.exists() and data_file.exists():
        try:
            if data_file.stat().st_size > 0:
                with open(data_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
        except (json.JSONDecodeError, OSError):
            return False
    return False

def save_response(number: int, data: dict, base_folder: str = "crawled_data") -> None:
    folder_path = Path(base_folder) / str(number)
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / "data.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Saved data to {file_path}")

class Crawler:
    def __init__(self, base_url: str, base_folder: str = "crawled_data", delay: float = 0.1):
        self.base_url = base_url
        self.base_folder = base_folder
        self.delay = delay
        self.successful_ids = set()
        self.checked_numbers = set()
        Path(base_folder).mkdir(parents=True, exist_ok=True)

    def try_url(self, number: int, target_name: str = "lux_ai_s3"):
        """Returns: (is_success, name_found)"""
        if number in self.checked_numbers:
            return False, ""

        self.checked_numbers.add(number)

        if is_id_already_downloaded(number, self.base_folder):
            self.successful_ids.add(number)
            return True, target_name

        url = self.base_url.format(number)

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                name = data.get("name", "")
                if name == target_name:
                    print(f"Success - Found {target_name}: {url}")
                    save_response(number, data, self.base_folder)
                    self.successful_ids.add(number)
                    return True, name
                else:
                    print(f"Different name found ({name}): {url}")
                    return False, name
            else:
                print(f"Failed ({response.status_code}): {url}")
                return False, ""

        except Exception as e:
            print(f"Error accessing {url}: {str(e)}")
            return False, ""

        finally:
            time.sleep(self.delay)

    def thoroughly_search_around(self, match_number: int, radius: int = 20):
        """
        When a match is found, thoroughly search all numbers around it.
        Stops in either direction when finding a non-lux match.
        """
        print(f"\nThoroughly searching around {match_number} (±{radius})")

        # Search upwards
        for offset in range(1, radius + 1):
            current = match_number + offset
            success, name = self.try_url(current)
            if success:
                print(f"Found additional match at {current}")
            if name and name != "lux_ai_s3":  # If we got any other name, stop searching upwards
                print(f"Found different name '{name}' at {current}, stopping upward search")
                break

        # Search downwards
        for offset in range(-1, -radius - 1, -1):
            current = match_number + offset
            success, name = self.try_url(current)
            if success:
                print(f"Found additional match at {current}")
            if name and name != "lux_ai_s3":  # If we got any other name, stop searching downwards
                print(f"Found different name '{name}' at {current}, stopping downward search")
                break



    def random_search_until_match(self, start_number: int, range_size: int = 100000, max_attempts: int = 100):
        """
        Randomly sample numbers until we find a match
        """
        lower_bound = max(0, start_number - range_size//2)
        upper_bound = start_number + range_size//2

        for attempt in range(max_attempts):
            number = random.randint(lower_bound, upper_bound)
            success, name = self.try_url(number)
            if success:
                print(f"\nFound match during random search at {number}")
                self.thoroughly_search_around(number)
                return number

            if attempt % 10 == 0:
                print(f"Random search attempt {attempt + 1}/{max_attempts}")

        return None

    def crawl(self, start_number: int):
        """
        Main crawling method that combines random search with thorough local search
        """
        current_number = start_number
        no_results_count = 0

        # First try the start number and search around it if it's a match
        success, name = self.try_url(current_number)
        if success:
            self.thoroughly_search_around(current_number)

        while no_results_count < 3:  # Stop after 3 complete failures
            print(f"\nStarting random search phase {no_results_count + 1}/3")

            # Try random search to find a new match
            found_number = self.random_search_until_match(current_number)

            if found_number:
                current_number = found_number
                no_results_count = 0
            else:
                no_results_count += 1
                print(f"No new matches found in attempt {no_results_count}/3")

        print("\nCrawling completed!")
        return sorted(list(self.successful_ids))

def create_index_file(successful_ids: List[int], base_folder: str = "crawled_data") -> None:
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
    START_NUMBER = 61294546
    BASE_FOLDER = "crawled_data"

    crawler = Crawler(
        base_url=BASE_URL,
        base_folder=BASE_FOLDER,
        delay=0.1
    )

    successful_ids = crawler.crawl(START_NUMBER)
    create_index_file(successful_ids, BASE_FOLDER)

    print(f"\nFinal Summary:")
    print(f"- Total successful IDs: {len(successful_ids)}")
    print(f"- Data saved in: {BASE_FOLDER}")

if __name__ == "__main__":
    main()
