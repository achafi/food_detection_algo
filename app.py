import io
import base64
import requests
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize FastAPI app
app = FastAPI()

# ==================== AUTO-DOWNLOAD YOLO MODEL ====================
MODEL_PATH = "models/yolov8_food.pt"
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO food detection model...")
    os.makedirs("models", exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    response = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            f.write(chunk)
    print("YOLO model downloaded!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# ==================== FASTAPI IMAGE UPLOAD ENDPOINT ====================
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Detects food items in the image, draws bounding boxes, and returns annotated image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLO model
    results = model(image)

    boxes_data = []
    detected_foods = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            class_name = result.names[cls]
            detected_foods.add(class_name)
            coords = box.xyxy[0].tolist()
            boxes_data.append({"class": class_name, "box": coords})

    if not detected_foods:
        return {"error": "No food detected"}

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    # Load font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    for item in boxes_data:
        x1, y1, x2, y2 = item['box']
        # box outline
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # label text
        text = item['class']
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # position label above box or below if at top
        y_text = y1 - text_height - 4 if y1 - text_height - 4 > 0 else y1 + 4
        text_pos = (x1, y_text)
        # background rectangle for readability
        draw.rectangle([
            text_pos,
            (text_pos[0] + text_width + 4, text_pos[1] + text_height + 4)
        ], fill="red")
        # draw text
        draw.text((text_pos[0] + 2, text_pos[1] + 2), text, fill="white", font=font)

    # Encode image to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Fetch nutrition info
    nutrition_info = {}

    return {
        "image": img_b64,
        "foods_detected": list(detected_foods),
        "boxes": boxes_data,
        "nutrition_info": nutrition_info
    }


def fetch_usda(food_name):
    """Fetch nutrition data from USDA API."""
    USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
    USDA_API_KEY = (
        "54tjbbR0SsWeESfETrxtmGRibk0qtO7etLWneN97"  # Replace with your own API key
    )

    try:
        params = {"query": food_name, "api_key": USDA_API_KEY}
        response = requests.get(USDA_API_URL, params=params)
        data = response.json()

        if "foods" in data and data["foods"]:
            nutrients = data["foods"][0]["foodNutrients"]
            return {
                "calories": find_nutrient(nutrients, 208),
                "protein": find_nutrient(nutrients, 203),
                "carbs": find_nutrient(nutrients, 205),
                "fat": find_nutrient(nutrients, 204),
            }
    except Exception:
        pass

    return None


def find_nutrient(nutrients, id):
    """Extract specific nutrient value from USDA response."""
    for nutrient in nutrients:
        if nutrient["nutrientId"] == id:
            return nutrient["value"]
    return "Unknown"


# ==================== RUN FASTAPI SERVER ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
