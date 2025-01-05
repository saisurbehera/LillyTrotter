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

import requests
import time
from typing import List, Dict
import json
import os
from pathlib import Path
import random

class Crawler:
    def __init__(self, base_url: str, base_folder: str = "crawled_data", delay: float = 0.5, target_count: int = 3000):
        self.base_url = base_url
        self.base_folder = base_folder
        self.delay = delay
        self.target_count = target_count
        self.successful_ids = set()
        self.checked_numbers = set()
        self.known_files_path = Path(base_folder) / "known_files.json"
        self.known_files = self.load_known_files()
        Path(base_folder).mkdir(parents=True, exist_ok=True)

    def load_known_files(self) -> Dict[str, List[int]]:
        """Load the record of known lux and non-lux files"""
        if self.known_files_path.exists():
            try:
                with open(self.known_files_path, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data.get('lux_files', []))} known lux files")
                    print(f"Loaded {len(data.get('non_lux_files', []))} known non-lux files")
                    return data
            except json.JSONDecodeError:
                pass
        return {"lux_files": [], "non_lux_files": []}

    def save_known_files(self):
        """Save the record of known files"""
        with open(self.known_files_path, 'w') as f:
            json.dump(self.known_files, f, indent=2)

    def is_known_non_lux(self, number: int) -> bool:
        """Check if we already know this is a non-lux file"""
        return number in self.known_files["non_lux_files"]

    def is_known_lux(self, number: int) -> bool:
        """Check if we already know this is a lux file"""
        return number in self.known_files["lux_files"]

    def try_url(self, number: int, target_name: str = "lux_ai_s3"):
        """Returns: (is_success, name_found)"""
        if number in self.checked_numbers:
            return False, ""

        self.checked_numbers.add(number)

        # Check known files first
        if self.is_known_non_lux(number):
            return False, "known_non_lux"
        if self.is_known_lux(number):
            self.successful_ids.add(number)
            return True, target_name

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
                    if number not in self.known_files["lux_files"]:
                        self.known_files["lux_files"].append(number)
                        self.save_known_files()
                    return True, name
                else:
                    print(f"Different name found ({name}): {url}")
                    if number not in self.known_files["non_lux_files"]:
                        self.known_files["non_lux_files"].append(number)
                        self.save_known_files()
                    return False, name
            else:
                print(f"Failed ({response.status_code}): {url}")
                return False, ""

        except Exception as e:
            print(f"Error accessing {url}: {str(e)}")
            return False, ""

        finally:
            time.sleep(self.delay)

    def thoroughly_search_around(self, match_number: int, radius: int = 100):
        """
        Smart search around a match using exponential step sizes to minimize hitting non-lux files.
        """
        if len(self.successful_ids) >= self.target_count:
            print(f"Target count of {self.target_count} reached!")
            return

        print(f"\nThoroughly searching around {match_number}")

        def search_direction(start: int, direction: int):
            if len(self.successful_ids) >= self.target_count:
                return

            step = 1
            current = start + direction
            last_success = start

            while step <= radius:
                if len(self.successful_ids) >= self.target_count:
                    return

                if self.is_known_non_lux(current):
                    print(f"Skipping known non-lux at {current}")
                    step *= 2
                    current = last_success + (direction * step)
                    continue

                success, name = self.try_url(current)

                if name == "lux_ai_s3":
                    print(f"Found lux match at {current}")
                    gap_start = min(last_success, current)
                    gap_end = max(last_success, current)
                    if gap_end - gap_start > 1:
                        for gap_num in range(gap_start + 1, gap_end):
                            if self.is_known_non_lux(gap_num):
                                continue
                            gap_success, gap_name = self.try_url(gap_num)
                            if gap_name != "lux_ai_s3" and gap_name != "":
                                break
                            if len(self.successful_ids) >= self.target_count:
                                return

                    last_success = current
                    current += direction
                    step = 1

                elif name != "":
                    print(f"Found different name '{name}' at {current}, stopping this direction")
                    break
                else:
                    step *= 2
                    current = last_success + (direction * step)

        search_direction(match_number, 1)
        search_direction(match_number, -1)

    def random_search_until_match(self, start_number: int, range_size: int = 1000000, max_attempts: int = 500):
        """
        Improved random search with larger range and target count check
        """
        if len(self.successful_ids) >= self.target_count:
            return None

        initial_range = 100
        current_range = initial_range

        for attempt in range(max_attempts):
            if len(self.successful_ids) >= self.target_count:
                return None

            if attempt % 10 == 0 and attempt > 0:
                current_range = min(current_range * 2, range_size)
                print(f"Expanding search range to ±{current_range}")

            lower_bound = max(0, start_number - current_range)
            upper_bound = start_number + current_range

            number = random.randint(lower_bound, upper_bound)

            if self.is_known_non_lux(number):
                continue

            success, name = self.try_url(number)

            if name == "lux_ai_s3":
                print(f"\nFound lux match during random search at {number}")
                self.thoroughly_search_around(number)
                return number

            if attempt % 10 == 0:
                print(f"Random search attempt {attempt + 1}/{max_attempts}")

        return None

    def crawl(self, start_number: int):
        """
        Main crawling method with target count limit
        """
        current_number = start_number
        no_results_count = 0

        success, name = self.try_url(current_number)
        if success:
            self.thoroughly_search_around(current_number)

        while no_results_count < 3 and len(self.successful_ids) < self.target_count:
            print(f"\nStarting random search phase {no_results_count + 1}/3")
            print(f"Current successful files: {len(self.successful_ids)}/{self.target_count}")

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
    START_NUMBER = 58306841
    BASE_FOLDER = "crawled_data"
    TARGET_COUNT = 3000

    crawler = Crawler(
        base_url=BASE_URL,
        base_folder=BASE_FOLDER,
        delay=0.5,
        target_count=TARGET_COUNT
    )

    successful_ids = crawler.crawl(START_NUMBER)
    create_index_file(successful_ids, BASE_FOLDER)

    print(f"\nFinal Summary:")
    print(f"- Total successful IDs: {len(successful_ids)}/{TARGET_COUNT}")
    print(f"- Data saved in: {BASE_FOLDER}")


if __name__ == "__main__":
    main()
