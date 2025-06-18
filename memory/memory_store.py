import json
import os

MEMORY_FILE = "memory_store.json"

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)
