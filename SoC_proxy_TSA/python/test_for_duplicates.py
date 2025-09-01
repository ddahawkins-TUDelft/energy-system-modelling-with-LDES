import os
import json
import hashlib
from collections import defaultdict

# Path to your folder
folder_path = "SoC_proxy_TSA/data/parameters"

# Dictionary to store hashes and corresponding filenames
hash_dict = defaultdict(list)

def file_hash(filepath):
    """Return a hash of the JSON content (ignoring formatting/whitespace)."""
    with open(filepath, "r") as f:
        data = json.load(f)  # load JSON so formatting differences don't matter
    # Serialize with sorted keys to ensure consistency
    normalized = json.dumps(data, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

# Loop through all .json files
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        h = file_hash(filepath)
        hash_dict[h].append(filename)

# Print duplicates
print("Duplicate files:")
for h, files in hash_dict.items():
    if len(files) > 1:
        print(" - " + ", ".join(files))
