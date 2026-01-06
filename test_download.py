from transformers import ViTImageProcessor, ViTForImageClassification

# REPLACE THIS WITH YOUR EXACT HUGGING FACE ID
MODEL_PATH = "harun-767/horse-breed-classifier" 

print(f"Attempting to download from: {MODEL_PATH}...")

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    print("✅ Processor downloaded successfully!")
    
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    print("✅ Model downloaded successfully!")
    
except Exception as e:
    print("\n❌ ERROR DETAILS:")
    print(e)