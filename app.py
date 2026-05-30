import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# DOWNLOAD MODEL IF NOT PRESENT
# =====================================

MODEL_PATH = "cnn_pneumonia_model.keras"

FILE_ID = "1GUO1D2xz8sw12BKyGwlSfNrNvrEcPoKi"

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading AI model... Please wait."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    color:#0E76A8;
    font-size:42px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================================
# HEADER
# =====================================

st.markdown(
    '<p class="title">🩺 AI Pneumonia Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based Chest X-Ray Analysis</p>',
    unsafe_allow_html=True
)

st.write("---")

# =====================================
# FILE UPLOADER
# =====================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing X-Ray..."):
        prediction = model.predict(img_array, verbose=0)[0][0]

    with col2:

        st.subheader("Prediction Result")

        if prediction > 0.5:

            confidence = prediction * 100

            st.error(
                f"🦠 Pneumonia Detected\n\nConfidence: {confidence:.2f}%"
            )

        else:

            confidence = (1 - prediction) * 100

            st.success(
                f"✅ Normal\n\nConfidence: {confidence:.2f}%"
            )

        st.progress(confidence / 100)

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

st.write("---")

st.info(
    "This AI system analyzes Chest X-Ray images and predicts whether pneumonia is present."
)