#!/usr/bin/env python3
"""
Script to download imgur images using HTTP requests with retry logic (no browser needed).
"""

import time
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def load_imgur_urls(json_file='imgur_images.json', txt_file='imgur_urls.txt'):
    """Load imgur URLs from existing files."""
    # Try JSON file first
    if Path(json_file).exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
        all_urls = []
        for urls in data.values():
            all_urls.extend(urls)
        return list(set(all_urls))

    # Fall back to txt file
    if Path(txt_file).exists():
        with open(txt_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    print("Error: No imgur URLs file found. Run scrape_imgur_images.py first.")
    return []


def download_image(url, output_path, max_retries=3, retry_delay=2):
    """Download a single image with retry logic."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://imgur.com/',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
    }

    for attempt in range(max_retries):
        try:
            # Create request with headers
            req = Request(url, headers=headers)

            # Download image
            with urlopen(req, timeout=30) as response:
                if response.status == 200:
                    # Check content type to ensure it's actually an image
                    content_type = response.headers.get('Content-Type', '')

                    if not content_type.startswith('image/'):
                        # Got HTML instead of image - likely region blocked
                        print(f"    Region blocked (got {content_type})")
                        return 'region_blocked'

                    image_data = response.read()

                    # Additional check: ensure data looks like an image (not HTML)
                    if image_data[:15].decode('utf-8', errors='ignore').strip().startswith('<!'):
                        print(f"    Region blocked (got HTML page)")
                        return 'region_blocked'

                    # Verify minimum size (imgur error pages are usually small)
                    if len(image_data) < 1000:
                        print(f"    Suspicious file size ({len(image_data)} bytes)")
                        return False

                    # Save the image
                    with open(output_path, 'wb') as f:
                        f.write(image_data)

                    return True
                else:
                    print(f"    HTTP {response.status}")

        except HTTPError as e:
            if e.code == 429:
                # Rate limited - wait longer
                wait_time = retry_delay * (attempt + 1) * 2
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif e.code in [403, 404]:
                # Don't retry for these
                print(f"    HTTP {e.code} - skipping")
                return False
            else:
                print(f"    HTTP {e.code}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        except URLError as e:
            print(f"    Connection error: {e.reason}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return False


def download_images(urls, download_dir='imgur_images', delay=3, max_retries=3):
    """Download all images with progress tracking."""
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)

    print(f"\n📥 Downloading {len(urls)} images...\n")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, url in enumerate(urls, 1):
        try:
            filename = url.split('/')[-1]
            filepath = download_path / filename

            # Skip if already exists
            if filepath.exists():
                print(f"[{i}/{len(urls)}] ⏭️  Skipped (exists): {filename}")
                skipped_count += 1
                continue

            print(f"[{i}/{len(urls)}] 🔄 Downloading: {filename}...", end=' ')

            if download_image(url, filepath, max_retries=max_retries):
                print(f"✅")
                success_count += 1
            else:
                print(f"❌")
                failed_count += 1

            # Add delay between requests
            if i < len(urls):
                time.sleep(delay)

        except Exception as e:
            print(f"[{i}/{len(urls)}] ❌ Failed: {filename} - {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Images saved to: {download_dir}/")
    print("=" * 60)


def main():
    """Main function."""
    print("=" * 60)
    print("IMGUR IMAGE DOWNLOADER (Simple HTTP)")
    print("=" * 60)

    # Load URLs
    print("\n📂 Loading imgur URLs...")
    urls = load_imgur_urls()

    if not urls:
        return

    print(f"✅ Loaded {len(urls)} unique URLs")

    # Ask for confirmation
    print("\nThis will download images using HTTP requests (no browser needed).")
    download = input("Continue? (y/n): ").strip().lower()

    if download != 'y':
        print("Cancelled.")
        return

    # Ask for delay
    try:
        delay_input = input("Delay between downloads in seconds (default=3, recommended for rate limiting): ").strip()
        delay = float(delay_input) if delay_input else 3.0
    except:
        delay = 3.0

    # Ask for max retries
    try:
        retries_input = input("Max retries per image (default=3): ").strip()
        max_retries = int(retries_input) if retries_input else 3
    except:
        max_retries = 3

    # Download images
    download_images(urls, delay=delay, max_retries=max_retries)


if __name__ == '__main__':
    main()