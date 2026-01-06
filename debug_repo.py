from huggingface_hub import list_repo_files

REPO_ID = "harun-767/horse-breed-classifier/vit-horse-model"

print(f"🔍 Looking inside: {REPO_ID}...")

try:
    files = list_repo_files(REPO_ID)
    print("\n✅ FILES FOUND:")
    for f in files:
        print(f" - {f}")
        
except Exception as e:
    print("\n❌ CRITICAL ERROR:")
    print(e)