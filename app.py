import streamlit as st
import tempfile
from PIL import Image
from model import predict


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DeepTrace | Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# CUSTOM CSS ONLY
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #18243d 0%,
                #0b1020 40%,
                #070b14 100%
            );
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .brand {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 30px;
    }

    .hero {
        text-align: center;
        margin-bottom: 45px;
    }

    .hero h1 {
        font-size: 46px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 17px;
        color: #aeb8cc;
    }

    .result-real {
        padding: 25px;
        border-radius: 18px;
        border: 2px solid #22c55e;
        background: rgba(34, 197, 94, 0.10);
        text-align: center;
    }

    .result-fake {
        padding: 25px;
        border-radius: 18px;
        border: 2px solid #ef4444;
        background: rgba(239, 68, 68, 0.10);
        text-align: center;
    }

    .result-title {
        font-size: 15px;
        color: #aeb8cc;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-value {
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .confidence {
        font-size: 19px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BRAND
# =========================================================

st.markdown(
    "🔍 **DeepTrace**"
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>Deepfake Detection System</h1>
        <p>
            Analyze images using an AI-powered detection model
            to determine whether an image is likely real or
            AI-generated / manipulated.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IMAGE + DETECTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        width="stretch"
    )

    detect_button = st.button(
        "🔍 Analyze Image",
        type="primary",
        width="stretch"
    )

    if detect_button:

        with st.spinner("Analyzing image..."):

            # SAME WORKING PIPELINE

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                image.save(temp_file.name)
                image_path = temp_file.name

            # SAME MODEL FUNCTION

            result = predict(image_path)


        # =================================================
        # RESULTS
        # =================================================

        prediction = result["prediction"]
        confidence = result["confidence"]
        real_probability = result["real_probability"]
        fake_probability = result["fake_probability"]


        st.divider()

        st.subheader("📊 Detection Result")


        # =================================================
        # REAL
        # =================================================

        if prediction.upper() == "REAL":

            st.success(
                f"🟢 REAL — Confidence: {confidence:.2f}%"
            )

            st.info(
                "The detector predicts that this image "
                "is likely a real photograph."
            )


        # =================================================
        # FAKE
        # =================================================

        else:

            st.error(
                f"🔴 FAKE — Confidence: {confidence:.2f}%"
            )

            st.warning(
                "The detector predicts that this image "
                "is likely AI-generated or manipulated."
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader("📈 Probability Analysis")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🟢 Real Probability",
                f"{real_probability:.2f}%"
            )

            st.progress(
                int(real_probability)
            )


        with col2:

            st.metric(
                "🔴 AI / Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.progress(
                int(fake_probability)
            )


        # =================================================
        # HOW IT WORKS
        # =================================================

        st.subheader("🧠 How It Works")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown("### 01 · Upload")

            st.write(
                "Upload an image in JPG, JPEG or PNG format."
            )


        with col2:

            st.markdown("### 02 · Analyze")

            st.write(
                "The image is processed and passed through "
                "the pretrained deepfake detection model."
            )


        with col3:

            st.markdown("### 03 · Result")

            st.write(
                "The system displays the predicted class "
                "along with confidence and probability."
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.divider()

        st.caption(
            "⚠️ Detection is probabilistic and may produce "
            "false results. The result should not be treated "
            "as absolute proof of authenticity."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DeepTrace · Deepfake Detection System · "
    "AI-assisted image authenticity analysis"
)