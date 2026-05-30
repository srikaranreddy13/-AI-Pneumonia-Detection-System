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
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "cnn_pneumonia_model.keras"
    )
    return model

model = load_model()

# =====================================
# HEADER
# =====================================

st.markdown(
    '<p class="title">🩺 AI Pneumonia Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Chest X-Ray Disease Detection using Deep Learning (CNN)</p>',
    unsafe_allow_html=True
)

st.write("---")

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    img = image.resize((150, 150))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)[0][0]

    with col2:

        st.subheader("Prediction Result")

        if prediction > 0.5:

            confidence = prediction * 100

            st.error(
                f"🦠 PNEUMONIA DETECTED\n\nConfidence: {confidence:.2f}%"
            )

        else:

            confidence = (1 - prediction) * 100

            st.success(
                f"✅ NORMAL\n\nConfidence: {confidence:.2f}%"
            )

        st.progress(float(confidence / 100))

        st.metric(
            "Prediction Score",
            f"{confidence:.2f}%"
        )

st.write("---")

st.info(
    "This AI model analyzes Chest X-Ray images and predicts whether Pneumonia is present."
)