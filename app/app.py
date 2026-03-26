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


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }

        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: white;
            border-radius: 24px;
            padding: 2rem 2rem 1.5rem 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 0;
        }

        /* 关键修复：直接给左右两列里的真实内容容器加卡片样式 */
        div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
            background: white;
            border-radius: 20px;
            padding: 1.25rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
            height: 100%;
        }

        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
            border: 1px solid #e5eefc;
            border-radius: 20px;
            padding: 1.2rem;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        .result-label {
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.95rem;
        }

        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stFileUploader"] section {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🐾 Animal Image Classifier</div>
            <p class="hero-subtitle">
                Upload an image and let the model predict whether it is a cat or a dog.
                Clean interface, instant result, and confidence breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Animal Image Classifier",
        page_icon="🐾",
        layout="wide",
    )

    inject_css()
    render_header()

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.subheader("Upload Image")
        st.caption("Supported formats: JPG, JPEG, PNG")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Upload an image to begin prediction.")

        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except UnidentifiedImageError:
                st.error("The uploaded file is not a valid image. Please upload a JPG or PNG file.")
                image = None
            except Exception:
                st.error("The image could not be opened. Please try a different file.")
                image = None

            if image is not None:
                st.image(image, caption="Uploaded image", use_container_width=True)

    with right_col:
        st.subheader("Prediction Result")

        if uploaded_file is None:
            st.info("Upload an image to see prediction results here 👀")
            return

        if "image" not in locals() or image is None:
            return

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
        confidence = float(result["confidence"]) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted class</div>
                <div class="result-value">{label}</div>
                <div class="small-muted">Model confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(confidence / 100, 0.0), 1.0))

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Prediction", label)
        with metric_col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        probabilities = result.get("probabilities", {})
        if probabilities:
            st.markdown("### Probability Breakdown")
            top_predictions = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for class_name, probability in top_predictions:
                pct = float(probability) * 100
                st.write(f"**{class_name.capitalize()}** — {pct:.1f}%")
                st.progress(min(max(float(probability), 0.0), 1.0))


if __name__ == "__main__":
    main()