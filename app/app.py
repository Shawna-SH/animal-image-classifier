from pathlib import Path
import sys

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import CLASS_NAMES, get_device, load_model, predict_image


CHECKPOINT_PATH = Path("artifacts/models/best_model.pth")


@st.cache_resource
def get_cached_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = get_device("auto")
    model = load_model(
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        num_classes=len(CLASS_NAMES),
    )
    return model, device


def main():
    st.set_page_config(page_title="Animal Image Classifier", page_icon="🐾")

    st.title("Animal Image Classifier")
    st.write("Upload an image to classify it as cat or dog")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG file.")
        return
    except Exception:
        st.error("The image could not be opened. Please try a different file.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model, device = get_cached_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("The model could not be loaded. Please check the checkpoint file.")
        return

    try:
        with st.spinner("Classifying image..."):
            result = predict_image(image_source=image, model=model, device=device)
    except Exception:
        st.error("Prediction failed. Please try another image.")
        return

    label = result["label"].capitalize()
    confidence = result["confidence"] * 100

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.1f}%")

    probabilities = result.get("probabilities", {})
    if probabilities:
        top_predictions = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]

        st.write("Top probabilities:")
        for class_name, probability in top_predictions:
            st.write(f"- {class_name.capitalize()}: {probability * 100:.1f}%")


if __name__ == "__main__":
    main()
