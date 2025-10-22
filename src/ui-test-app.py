# streamlit_app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Onboarding & Data Quality Agent", layout="wide")
st.title("🧭 Onboarding & Data Quality Agent")

# --- Sidebar Progress Steps ---
steps = ["Upload", "Semantic Mapping", "Data Quality", "Rules / Teaching", "Review & Export"]
st.markdown("### 1) Upload")

# --- Left Column: Schema ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Schema")
    schema = st.selectbox("Select predefined schema", ["med_claims"])
    st.markdown(
        """
        **fields:**
        - clm_id  
        - subscriber_id  
        - svc_from  
        - svc_to  
        - paid_date  
        - bill_npl  
        - rend_npl  
        - pos  
        - cpt  
        - mod1  
        - dsl  
        - unlts  
        - allowed  
        - plan_paid  
        - member_resp
        """
    )
    st.success("Loaded 18 fields from schema 'med_claims'")

# --- Right Column: Dataset ---
with col2:
    st.subheader("Dataset (CSV)")
    use_demo = st.checkbox("Use demo dataset", value=True)

    uploaded = None
    if not use_demo:
        uploaded = st.file_uploader("Upload file", type=["csv"])

    # Demo dataset
    data = pd.DataFrame({
        "clm_id": ["00187910","00288752","00518512","00626720","00518512","00518512","00519515"],
        "subscriber_id": ["002337320","003468020","005185129","006257208","004948123","004948123","004101034"],
        "svc": [2.1,2.1,2.1,50,23,55,37],
        "paid_date": [10225718,10225718,10229718,10230771,10235718,10230771,10235718],
        "bill_npi": [455,455,455,452,570,567,425],
        "rend_npi": [6092,6092,6092,3300,4200,4200,3760],
        "proc": [281,128,260,147,520,340,326]
    })

    df = pd.read_csv(uploaded) if uploaded is not None else data
    st.dataframe(df, use_container_width=True)
    st.caption("%t 0   Null %   Unique %")

st.markdown("✅ **Next step:** Go to ‘Semantic Mapping’")
