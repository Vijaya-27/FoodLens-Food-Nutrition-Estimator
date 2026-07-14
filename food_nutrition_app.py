import os
import json
from typing import Dict, Tuple

import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, models

APP_TITLE = "Food Nutrition Detection (Deep Learning)"
MODEL_PATH = "nutrition_model.keras"

OUTPUT_KEYS = [
    "calories",
    "protein",
    "carbs",
    "fat",
]

# Mapping from CSV columns to model output order.
# Your CSV contains: calories, protein, carbs, fat, iron, vitamin_c, ...
CSV_FEATURE_KEY_ORDER = OUTPUT_KEYS



def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(len(OUTPUT_KEYS), activation="linear", name="nutrition")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="nutrition_regressor")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


DATASET_CSV_PATH = "Food_Nutrition_Dataset.csv"


def load_dataset_csv(csv_path: str = DATASET_CSV_PATH):
    """Load nutrition targets from CSV.

    NOTE: The provided CSV appears to contain nutrition values but not image paths.
    This function therefore trains a demo model using *synthetic images* and *real nutrition targets*.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Drop rows with missing target values.
    for k in OUTPUT_KEYS:
        if k not in df.columns:
            raise ValueError(f"CSV missing required column: {k}")

    df = df.dropna(subset=OUTPUT_KEYS).reset_index(drop=True)

    y = df[OUTPUT_KEYS].astype(np.float32).values
    y = np.clip(y, 0, None)

    # Create synthetic image inputs as placeholders.
    # Without image paths, we cannot learn image->nutrition mapping from this CSV alone.
    rng = np.random.default_rng(42)
    img_size = (128, 128, 3)
    X = rng.integers(0, 256, size=(len(y), *img_size), dtype=np.uint8)

    return X, y


def generate_synthetic_dataset(n: int = 800, img_size=(128, 128, 3), seed: int = 42):
    """Fallback synthetic dataset if CSV is missing."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, size=(n, *img_size), dtype=np.uint8)

    brightness = X.mean(axis=(1, 2, 3)) / 255.0

    calories = 50 + 350 * brightness + rng.normal(0, 15, size=n)
    protein = 1 + 25 * brightness + rng.normal(0, 1.5, size=n)
    carbs = 5 + 60 * brightness + rng.normal(0, 3.0, size=n)
    fat = 0.5 + 40 * brightness + rng.normal(0, 2.5, size=n)

    y = np.stack([calories, protein, carbs, fat], axis=1).astype(np.float32)
    y = np.clip(y, 0, None)
    return X, y



def normalize_targets(y: np.ndarray):
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True) + 1e-6
    yn = (y - mean) / std
    return yn, mean.astype(np.float32), std.astype(np.float32)


def denormalize_predictions(pred_norm: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return pred_norm * std + mean


def ensure_model(trainer_mode: str = "train_if_missing"):
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    if trainer_mode == "never_train":
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Enable training in the UI to generate one (demo uses synthetic data)."
        )

    st.warning(
        "No trained model found. Training a DEMO model on synthetic data. "
        "For real nutrition accuracy, replace synthetic training with a real dataset."
    )

    model = build_model()

    # Prefer training from the provided CSV.
    try:
        X, y = load_dataset_csv(DATASET_CSV_PATH)
        if len(y) < 50:
            raise ValueError("CSV too small to train")
    except Exception as e:
        st.warning(f"Could not use CSV for training ({e}). Falling back to synthetic dataset.")
        X, y = generate_synthetic_dataset(n=800)

    y_norm, mean, std = normalize_targets(y)


    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.85 * len(X))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr = X[tr_idx].astype(np.float32)
    y_tr = y_norm[tr_idx]
    X_va = X[va_idx].astype(np.float32)
    y_va = y_norm[va_idx]

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=8,
        batch_size=32,
        verbose=1,
    )
    model.save(MODEL_PATH)

    norm_path = MODEL_PATH + ".norm.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.flatten().tolist(), "std": std.flatten().tolist()}, f)

    return model


def load_norm_params():
    norm_path = MODEL_PATH + ".norm.json"
    if not os.path.exists(norm_path):
        return None

    with open(norm_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    mean = np.array(d["mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(d["std"], dtype=np.float32).reshape(1, -1)
    return mean, std


def predict_nutrition(model: tf.keras.Model, pil_image) -> Dict[str, float]:
    from PIL import Image

    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image)

    img = pil_image.convert("RGB").resize((128, 128))
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0).astype(np.float32)

    pred = model.predict(x, verbose=0)

    norm = load_norm_params()
    if norm is not None:
        mean, std = norm
        pred = denormalize_predictions(pred, mean, std)

    pred = pred.reshape(-1)
    return {k: float(v) for k, v in zip(OUTPUT_KEYS, pred.tolist())}


seed_everything(42)

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Model")
    st.caption("Deep learning inference runs locally via TensorFlow. Streamlit is the UI.")

    train_option = st.selectbox(
        "Model handling",
        options=["train_if_missing", "never_train"],
        index=0,
        help="If model doesn't exist, Streamlit will train a DEMO model on synthetic data.",
    )

    if st.button("Load/Train Model", type="primary"):
        with st.spinner("Loading or training model..."):
            m = ensure_model(train_option)
            st.session_state["model"] = m
            st.session_state["model_loaded"] = True
        st.success("Model ready")

    if "model_loaded" not in st.session_state:
        st.session_state["model_loaded"] = False

if not st.session_state["model_loaded"]:
    st.info("Click **Load/Train Model** in the sidebar to initialize the deep learning model.")


uploaded = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded is not None:
    from PIL import Image

    pil_image = Image.open(uploaded)
    st.image(pil_image, caption="Uploaded image", use_column_width=True)

    if st.button("Detect Nutrition", type="primary", disabled=not st.session_state.get("model_loaded", False)):
        with st.spinner("Running deep learning inference..."):
            model = st.session_state["model"]
            results = predict_nutrition(model, pil_image)

        st.subheader("Predicted nutrition (per 100g)")

        # Use correct unit for current OUTPUT_KEYS
        # calories -> kcal, protein/carbs/fat -> g


        # Pretty layout
        col1, col2 = st.columns(2)
        keys = list(results.keys())
        for i, k in enumerate(keys):
            v = results[k]
            unit = "kcal" if k == "calories" else "g"
            label = k.replace("_", " ")

            if i % 2 == 0:
                col1.metric(label, f"{v:.1f} {unit}")
            else:
                col2.metric(label, f"{v:.1f} {unit}")

        st.caption("Demo note: Unless you replace training with a real nutrition dataset, predictions are not accurate.")


