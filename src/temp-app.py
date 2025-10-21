# streamlit_app.py
import os, json, glob
import streamlit as st
from pathlib import Path
placeholder = st.empty()

# 1) Use the SAME dir as the producer. Make it absolute.
BASE_DIR = Path(__file__).resolve().parent
# Go one directory up, then into 'review_queue'
REVIEW_DIR = (BASE_DIR.parent / "review_queue").resolve()


st.title("RAG Mapping – Human Review")

# 2) Debug panel
st.caption(f"cwd: {os.getcwd()}")
st.caption(f"REVIEW_DIR: {REVIEW_DIR}")
st.caption(f"exists: {os.path.exists(REVIEW_DIR)}  isdir: {os.path.isdir(REVIEW_DIR)}")

print("Looking at review path - ")
print(BASE_DIR)

print("Looking at path - ")
print(REVIEW_DIR)

# 3) Autorefresh so new files appear
if st.button("Refresh"):
    st.rerun()

# 4) Enumerate jsons safely
files = sorted(glob.glob(os.path.join(REVIEW_DIR, "*.json")))
st.caption(f"found json files: {len(files)}")
if len(files) > 0:
    with st.expander("Show filenames"):
        for p in files:
            st.write(os.path.basename(p))

if not files:
    st.info("No pending reviews found in REVIEW_DIR.")
    st.stop()

# 5) Select a file
path = st.selectbox("Pending review:", files, format_func=os.path.basename)

# 6) Load with error handling (skip corrupt/empty files)
payload = None
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception as e:
    st.error(f"Failed to parse {os.path.basename(path)}: {e}")
    if st.button("Delete corrupt file"):
        os.remove(path)
    st.stop()

# 7) Render suggestions
st.subheader(f"Review ID: {payload.get('review_id','<none>')}")
suggestions = payload.get("suggestions", {})

if not suggestions:
    st.warning("No suggestions in payload.")
    st.stop()

decisions = {}
for col, suggs in suggestions.items():
    st.markdown(f"### {col}")
    # guard against non-list content
    if not isinstance(suggs, list) or not suggs:
        st.warning("No suggestions for this column.")
        decisions[col] = None
        continue
    labels = [
        f"{s.get('target','?')}  (score={s.get('score',0):.2f}, sim={s.get('retrieval_sim',0):.2f})"
        for s in suggs
    ]
    choice = st.radio("Pick target or reject", labels + ["<reject>"], key=f"opt-{col}", horizontal=True)
    decisions[col] = None if choice == "<reject>" else suggs[labels.index(choice)]["target"]

if st.button("Submit decisions"):
    out = {
        "review_id": payload.get("review_id"),
        "decisions": decisions,
        "approved": True,
    }
    out_path = os.path.join(REVIEW_DIR, f"{payload.get('review_id','review')}.decision.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    st.success(f"Saved: {out_path}")
