import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


st.title("AI Image Detection System")

st.write(
    "Upload an image to check whether it is likely real or AI-generated."
)


@st.cache_resource
def load_model():

    model_name = "capcheck/ai-image-detection"

    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name
    )

    return processor, model


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect AI Image"):

        with st.spinner("Analyzing image..."):

            processor, model = load_model()

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():

                outputs = model(**inputs)

            probabilities = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )[0]

            real_probability = (
                probabilities[0].item() * 100
            )

            fake_probability = (
                probabilities[1].item() * 100
            )


        st.subheader("Detection Result")

        st.write(
            f"Real Probability: **{real_probability:.2f}%**"
        )

        st.write(
            f"AI Probability: **{fake_probability:.2f}%**"
        )


        if fake_probability >= 75:

            st.error("🔴 Likely AI Generated")

            st.info(
                "The detector has high confidence that "
                "this image may be AI-generated."
            )

        elif real_probability >= 75:

            st.success("🟢 Likely Real")

            st.info(
                "The detector has high confidence that "
                "this image may be a real photograph."
            )

        else:

            st.warning("🟡 Uncertain")

            st.info(
                "The detector is not confident enough "
                "to classify this image reliably."
            )


        st.caption(
            "Note: AI image detection is probabilistic "
            "and can produce false results."
        )