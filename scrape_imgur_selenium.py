#!/usr/bin/env python3
"""
Script to download imgur images using SeleniumBase (browser automation) to avoid rate limiting.
"""

import time
import json
import base64
from pathlib import Path
from seleniumbase import SB


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


def download_image_with_seleniumbase(sb, url, output_path):
    """Download a single image using SeleniumBase."""
    try:
        # Navigate to the image URL
        sb.open(url)

        # Wait for page to load
        sb.sleep(0.5)

        # Get the image using JavaScript to convert to base64
        script = """
        async function getImageAsBase64(url) {
            try {
                const response = await fetch(url);
                const blob = await response.blob();
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
            } catch(e) {
                return null;
            }
        }
        return await getImageAsBase64(arguments[0]);
        """

        # Execute the script to get base64 data
        base64_data = sb.execute_async_script(script, url)

        if base64_data and base64_data.startswith('data:image'):
            # Remove the data URL prefix
            image_data = base64_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

            # Save the image
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

            return True

        return False

    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_images_seleniumbase(urls, download_dir='imgur_images', headless=True, delay=2):
    """Download all images using SeleniumBase."""
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)

    print(f"\n📥 Setting up browser with SeleniumBase...")

    # Use SeleniumBase context manager
    with SB(uc=True, headless=headless, ad_block=True) as sb:
        print(f"📥 Downloading {len(urls)} images with browser automation...\n")

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

                if download_image_with_seleniumbase(sb, url, filepath):
                    print(f"✅")
                    success_count += 1
                else:
                    print(f"❌")
                    failed_count += 1

                # Add delay between requests to be polite
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

    print("\n🔚 Browser closed.")


def main():
    """Main function."""
    print("=" * 60)
    print("IMGUR IMAGE DOWNLOADER (SELENIUMBASE)")
    print("=" * 60)

    # Load URLs
    print("\n📂 Loading imgur URLs...")
    urls = load_imgur_urls()

    if not urls:
        return

    print(f"✅ Loaded {len(urls)} unique URLs")

    # Ask for confirmation
    print("\nThis will use browser automation (SeleniumBase) to download images.")
    download = input("Continue? (y/n): ").strip().lower()

    if download != 'y':
        print("Cancelled.")
        return

    # Ask if headless
    headless_input = input("Run in headless mode (no visible browser)? (y/n, default=y): ").strip().lower()
    headless = headless_input != 'n'

    # Ask for delay
    try:
        delay_input = input("Delay between downloads in seconds (default=2): ").strip()
        delay = float(delay_input) if delay_input else 2.0
    except:
        delay = 2.0

    # Download images
    download_images_seleniumbase(urls, headless=headless, delay=delay)


if __name__ == '__main__':
    main()
