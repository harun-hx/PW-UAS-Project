import base64
import json
import os

# === CONFIGURATION ===
# Change this to the path of the image you want to test
IMAGE_PATH = r"C:\Users\Harun\Documents\Web\ProjectNateHiggerson\horse.jpg"
OUTPUT_FILE = "payload.json"

def create_payload():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Processing image: {IMAGE_PATH}...")

    with open(IMAGE_PATH, "rb") as image_file:
        # 1. Read binary file
        binary_data = image_file.read()
        
        # 2. Encode to Base64 string
        base64_encoded = base64.b64encode(binary_data).decode('utf-8')
        
        # 3. Create the JSON structure
        payload = {
            "image": base64_encoded
        }

    # 4. Save to file (because printing it to console is too messy)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f)

    print(f"✅ Success! JSON payload saved to '{OUTPUT_FILE}'")
    print("👉 Open that file, copy ALL text, and paste it into Postman's Body (Raw > JSON).")

if __name__ == "__main__":
    create_payload()