#!/usr/bin/env python3
import json
import time
import requests
from pathlib import Path

def download_image(url, output_path):
    """Download a single image using requests."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Load mapping
with open('imgur_download_map.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("DOWNLOADING IMGUR IMAGES TO GRAPHICS FOLDERS")
print("=" * 60)

total = sum(len(post['urls']) for post in data['post_assignments'].values())
print(f"\nTotal images to download: {total}\n")

success = 0
failed = 0
skipped = 0
count = 0

for post_name, post_data in data['post_assignments'].items():
    folder = Path(post_data['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 {post_name} ({post_data['post_file']})")
    
    for url in post_data['urls']:
        count += 1
        filename = url.split('/')[-1]
        filepath = folder / filename
        
        if filepath.exists():
            print(f"  [{count}/{total}] ⏭️  {filename} (exists)")
            skipped += 1
            continue
        
        print(f"  [{count}/{total}] 🔄 {filename}...", end=' ')
        
        if download_image(url, filepath):
            print("✅")
            success += 1
        else:
            print("❌")
            failed += 1
        
        time.sleep(1)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Success: {success}")
print(f"⏭️  Skipped: {skipped}")
print(f"❌ Failed: {failed}")
print("=" * 60)
