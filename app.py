import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Tampering Detection", layout="centered")

st.title("🛡️ Image Tampering Detection Tool")
st.write("Upload an image to detect edited regions")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect_tampering(image):
    output = image.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Step 4: Thresholding
    _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Step 5: Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            detected = True

    return output, detected

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result, detected = detect_tampering(image)

    st.subheader("Result")
    st.image(result, channels="BGR")

    if detected:
        st.error("⚠️ Edited / Tampered Region Detected")
    else:
        st.success("✅ No Significant Tampering Detected")
