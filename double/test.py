import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("best_model.keras")

st.title("Double Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(image,caption="Original Image")

    blur = cv2.GaussianBlur(image,(5,5),0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = np.ones((2,2),np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.resize(thresh,(128,64))

    st.image(thresh,caption="Processed Image")

    thresh = thresh.astype("float32")/255.0

    thresh = thresh.reshape(1,64,128,1)

    prediction = model.predict(thresh,verbose=0)

    predicted = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(f"Predicted Number : {predicted:02d}")

    st.write(f"Confidence : {confidence*100:.2f}%")