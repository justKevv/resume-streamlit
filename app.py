"""Resume Role Predictor — fully local, no API.

Upload a resume PDF -> extract text -> predict role with models copied
from resume-api (models/*.pkl + logic in src/predictor.py, text utils in
src/text.py).

Run (fish + uv):
  source venv/bin/activate.fish
  uv pip install -r requirements.txt
  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
from pypdf import PdfReader

from src.predictor import get_category_prediction

st.set_page_config(page_title="Resume Role Predictor", page_icon="📄", layout="centered")


def extract_pdf_text(uploaded_file) -> tuple[str, int]:
    reader = PdfReader(uploaded_file)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages).strip(), len(reader.pages)


# ---- model status ----
with st.sidebar:
    st.header("🧠 Local model")
    st.caption("`models/` copied from `resume-api/models/`.\nLogic copied from `app/core/models.py`, `app/services/ranking.py`, `app/utils/text.py`.")
    try:
        from src.predictor import list_categories

        cats = list_categories()
        st.success(f"Model loaded — {len(cats)} categories")
        with st.expander("Known categories"):
            st.write(cats)
    except Exception as e:
        st.error(f"Model NOT loaded:\n\n{e}")

st.title("📄 Resume Role Predictor")
st.caption("Upload a resume PDF — text is extracted and the role is predicted locally.")

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])

pdf_text, n_pages = "", 0
if uploaded is not None:
    try:
        pdf_text, n_pages = extract_pdf_text(uploaded)
        if not pdf_text:
            st.warning("No extractable text found in this PDF (it may be scanned images).")
        else:
            st.success(f"Extracted {len(pdf_text)} characters from {n_pages} page(s).")
    except Exception as e:
        st.error(f"Could not read PDF: {e}")

profile_text = st.text_area(
    "Resume text (auto-filled from PDF, editable)",
    value=pdf_text,
    height=260,
    placeholder="Upload a PDF above, or paste resume text here…",
)

if st.button("🔮 Predict role", type="primary", use_container_width=True):
    if not profile_text.strip():
        st.warning("Please upload a PDF or enter resume text first.")
    else:
        with st.spinner("Predicting…"):
            try:
                category = get_category_prediction(profile_text)
                st.success("Done!")
                st.markdown(f"### 🎯 Predicted role: **{category}**")
            except (FileNotFoundError, RuntimeError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with st.expander("Preview extracted text"):
    st.write(profile_text if profile_text else "_No text yet._")
