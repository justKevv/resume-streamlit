# Resume Role Predictor — fully local (no API)

`app.py` uploads a resume **PDF**, extracts text, and predicts the role locally.
Model + code copied from `../resume-api` (no network calls):

| This folder | Copied from resume-api |
|---|---|
| `models/*.pkl` | `resume-api/models/*.pkl` |
| `src/text.py` | `resume-api/app/utils/text.py` (`clean_resume`) |
| `src/predictor.py` | `resume-api/app/core/models.py` + `app/services/ranking.py::get_category_prediction` (classification part only; sentence-transformers / geo ranking omitted) |

> Note: the `.pkl` files in `resume-api/models/` are currently Git LFS
> pointers, so they were copied as pointers. Run `git lfs pull` inside
> `resume-api` (install `git-lfs` first), then re-copy `models/*.pkl`
> here to get working binaries. The app shows a clear error until then.

## Setup (fish + venv + uv)

```fish
cd /Users/admin/Documents/coding/python/resume
source venv/bin/activate.fish
uv pip install -r requirements.txt
```

## Run

```fish
source venv/bin/activate.fish
streamlit run app.py
```

Then open http://localhost:8501.
