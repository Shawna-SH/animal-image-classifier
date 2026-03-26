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
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: clamp(0.9rem, 2vw, 1.5rem);
            padding-right: clamp(0.9rem, 2vw, 1.5rem);
        }

        .hero-card {
            background: white;
            border-radius: 24px;
            padding: clamp(1.1rem, 3vw, 2rem);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: clamp(1.55rem, 4vw, 2.2rem);
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0.4rem;
            line-height: 1.1;
            word-break: break-word;
        }

        .hero-subtitle {
            font-size: clamp(0.95rem, 2vw, 1.05rem);
            color: #6b7280;
            margin-bottom: 0;
            line-height: 1.55;
        }

        div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
            background: white;
            border-radius: 20px;
            padding: clamp(0.95rem, 2.4vw, 1.25rem);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
            height: 100%;
        }

        .insight-card {
            background: #f9fbff;
            border: 1px solid #e6eefb;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0;
        }

        .insight-title {
            font-size: clamp(0.95rem, 1.6vw, 1rem);
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.35rem;
            line-height: 1.35;
        }

        .insight-text {
            font-size: clamp(0.95rem, 1.8vw, 1.05rem);
            color: #4b5563;
            margin: 0;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }

        .panel-section-title {
            font-size: clamp(1.25rem, 3vw, 1.55rem);
            font-weight: 800;
            color: #1f2937;
            margin: 1.1rem 0 0.9rem 0;
            line-height: 1.15;
            word-break: break-word;
        }

        .panel-section-tight {
            margin-top: 0.75rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: clamp(0.9rem, 2vw, 1.8rem);
            margin-top: 1rem;
            margin-bottom: 0.6rem;
        }

        .metric-card {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-start;
            gap: 0.75rem;
            min-width: 0;
        }

        .metric-label {
            color: #2f3747;
            font-weight: 500;
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.15;
            font-size: clamp(1rem, 2vw, 1.2rem);
            min-width: 0;
        }

        .metric-value {
            color: #1f2432;
            line-height: 0.9;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-size: clamp(1.8rem, 4vw, 3.2rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            min-width: 0;
            max-width: 100%;
        }

        .metric-subvalue {
            color: #0f9f5a;
            background: #e8f6ee;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            width: fit-content;
            max-width: 100%;
            padding: 0.3rem 0.7rem;
            font-size: clamp(0.85rem, 1.6vw, 0.95rem);
            font-weight: 700;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .probability-label-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.35rem;
            color: #374151;
            font-size: clamp(0.95rem, 1.8vw, 1rem);
            font-weight: 700;
            flex-wrap: wrap;
        }

        .probability-name {
            color: #1f2937;
            overflow-wrap: anywhere;
        }

        .probability-value {
            color: #4b5563;
            font-weight: 700;
        }

        .footnote {
            color: #6b7280;
            font-size: clamp(0.88rem, 1.7vw, 0.95rem);
            line-height: 1.5;
            margin-top: 0.85rem;
            overflow-wrap: anywhere;
        }

        @media (max-width: 1100px) {
            .main .block-container {
                max-width: 980px;
            }
        }

        @media (max-width: 900px) {
            .panel-section-title {
                margin-top: 1rem;
                margin-bottom: 0.75rem;
            }
        }

        @media (max-width: 768px) {
            .hero-card {
                border-radius: 20px;
            }

            div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
                border-radius: 18px;
            }

            .insight-card,
            .metric-card {
                border-radius: 16px;
            }

            .metric-card {
                min-height: auto;
            }
        }

        @media (max-width: 640px) {
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }

            .hero-title {
                max-width: 14ch;
            }

            .metric-grid {
                grid-template-columns: 1fr;
                gap: 0.75rem;
            }

            .metric-value {
                font-size: clamp(1.4rem, 6vw, 2.2rem);
            }

            .probability-label-row {
                gap: 0.4rem;
                margin-bottom: 0.25rem;
            }
        }

        @media (max-width: 520px) {
            .metric-label {
                font-size: 0.95rem;
            }

            .metric-value {
                font-size: clamp(1.25rem, 7vw, 1.9rem);
            }

            .panel-section-title {
                font-size: 1.15rem;
            }
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


def build_result_summary(result: dict) -> dict:
    probabilities = result.get("probabilities", {})
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    top_label, top_prob = ranked[0]
    runner_up_label, runner_up_prob = ranked[1]
    margin = top_prob - runner_up_prob
    if margin >= 0.5:
        reading = (
            f"This is a clear separation: the model sees stronger visual evidence for {top_label} "
            f"than for {runner_up_label}."
        )
    elif margin >= 0.2:
        reading = (
            f"This is a moderate separation: {top_label} is favored, but {runner_up_label} still has "
            "meaningful support."
        )
    else:
        reading = (
            f"This is a close call: the model sees similar evidence for {top_label} and {runner_up_label}."
        )

    caveat = (
        "This model only chooses between cat and dog, so unusual images, mixed scenes, or low-quality photos "
        "can still produce confident-looking predictions."
    )

    return {
        "top_label": top_label,
        "top_prob": top_prob,
        "runner_up_label": runner_up_label,
        "runner_up_prob": runner_up_prob,
        "margin": margin,
        "reading": reading,
        "caveat": caveat,
        "ranked_probabilities": ranked,
    }


def render_metric_cards(label: str, runner_up: str, runner_up_prob: float, margin: float):
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">{label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Runner-up</div>
                <div class="metric-value">{runner_up}</div>
                <div class="metric-subvalue">↑ {runner_up_prob:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Decision margin</div>
                <div class="metric-value">{margin:.1f} pts</div>
            </div>
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

    left_col, right_col = st.columns([1.05, 1], gap="medium")

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
                width, height = image.size
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("Image size", f"{width}×{height}")
                with info_col2:
                    st.metric("Color mode", image.mode)

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

        summary = build_result_summary(result)
        label = summary["top_label"].capitalize()
        runner_up = summary["runner_up_label"].capitalize()
        runner_up_prob = float(summary["runner_up_prob"]) * 100
        margin = float(summary["margin"]) * 100

        st.markdown(
            '<div class="panel-section-tight"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">How to read this result</div>
                <p class="insight-text">{summary["reading"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_metric_cards(
            label=label,
            runner_up=runner_up,
            runner_up_prob=runner_up_prob,
            margin=margin,
        )

        st.markdown(
            '<div class="panel-section-title">Probability Breakdown</div>',
            unsafe_allow_html=True,
        )
        for class_name, probability in summary["ranked_probabilities"]:
            pct = float(probability) * 100
            st.markdown(
                f"""
                <div class="probability-label-row">
                    <span class="probability-name">{class_name.capitalize()}</span>
                    <span class="probability-value">{pct:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(float(probability), 0.0), 1.0))

        st.markdown(
            '<div class="panel-section-title">Model Scope</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-text">{summary["caveat"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="footnote">
                Decision margin is the gap between the top prediction and the runner-up.
                Larger gaps usually indicate a clearer separation between the two classes.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
