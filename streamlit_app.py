import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Food Detection App", layout="centered")

st.title("🍽️ AI-Based Food Recognition")
st.markdown("Upload a food image and detect food items using YOLOv8.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    placeholder = st.empty()
    placeholder.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting food items..."):
        # Send image to FastAPI backend
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post("http://localhost:8000/upload/", files=files)

        if response.status_code == 200:
            result = response.json()
            # Display annotated image with bounding boxes
            annotated_b64 = result.get("image")
            if annotated_b64:
                decoded_bytes = base64.b64decode(annotated_b64)
                annotated_img = Image.open(io.BytesIO(decoded_bytes))
                placeholder.image(annotated_img, caption="Annotated Image", use_column_width=True)
            detected = result.get("foods_detected", [])
            nutrition = result.get("nutrition_info", {})

            st.success("Detection complete!")
            st.subheader("🍱 Detected Items")
            st.write(", ".join(detected) if detected else "No food items detected.")

            if nutrition:
                st.subheader("📊 Nutrition Info")
                for item, data in nutrition.items():
                    st.markdown(f"**{item.capitalize()}**")
                    st.write(data)
            else:
                st.info("No nutritional data found.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
