"""Local model loading + category prediction.

Copied from resume-api:
  - app/core/models.py  (TF-IDF + label encoder + random forest loading only;
                         sentence-transformers / geo ranking intentionally omitted)
  - app/services/ranking.py :: get_category_prediction (logic kept 1:1)

Models live in <resume>/models/*.pkl (copied from resume-api/models/).
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(60).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_models():
    paths = {
        "tfidf": os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        "label_encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "random_forest": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model files in {MODEL_DIR}: {missing}")
    pointers = [k for k, p in paths.items() if _is_lfs_pointer(p)]
    if pointers:
        raise RuntimeError(
            "Model files are Git LFS pointers, not real binaries "
            f"({', '.join(pointers)} in {MODEL_DIR}). "
            "Run `git lfs pull` inside resume-api (needs git-lfs installed), "
            "then copy models/*.pkl over again."
        )
    tfidf_vectorizer = joblib.load(paths["tfidf"])
    le = joblib.load(paths["label_encoder"])
    rf_model = joblib.load(paths["random_forest"])
    return tfidf_vectorizer, le, rf_model


def get_category_prediction(profile_text: str) -> str:
    """Predict job category — same logic as resume-api's ranking.get_category_prediction."""
    tfidf_vectorizer, le, rf_model = _load_models()
    # NOTE: resume-api computes clean_resume() then discards it and uses
    # profile_text.lower(). Kept 1:1 so local predictions match the API.
    cleaned_text = profile_text.lower()
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction_encoded = rf_model.predict(vectorized_text)[0]
    category = le.inverse_transform([prediction_encoded])[0]
    return str(category)


def list_categories() -> list[str]:
    """All known labels from the copied label encoder."""
    _, le, _ = _load_models()
    return [str(c) for c in le.classes_]
