import hashlib
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sqlalchemy import create_engine, text

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise EHR & Clearinghouse Integration",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 28px; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 16px; font-weight: 500; color: #4B5563; margin-top: 5px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 8px; }
    .stAlert { padding: 10px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">🛡️ PrimEra: EHR (Epic/Meditech) & Clearinghouse AI Integration Prototype</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">FHIR API interception (simulated), claims denial-risk scoring, payer policy '
    'reference lookup, and an automated appeal-letter draft generator.</p>',
    unsafe_allow_html=True,
)
st.info(
    "🔬 **Prototype notice:** The denial-risk model below is trained on synthetic/simulated claims "
    "data to demonstrate the ML pipeline end-to-end. It is not trained on real historical claims "
    "outcomes, so its probabilities are illustrative, not clinically or actuarially validated. "
    "The EHR connection panel similarly runs in simulation/offline mode unless a live OAuth endpoint "
    "and credentials are supplied.",
    icon="🔬",
)
st.markdown("---")

# --- Configuration ---
DB_CONNECTION_STRING = os.getenv("RCM_DATABASE_URL", "sqlite:///rcm_production_master.db")
EPIC_TOKEN_ENDPOINT = os.getenv(
    "EPIC_TOKEN_ENDPOINT", "https://fhir.epic.com/interconnect-fhir-oauth/api/oauth2/token"
)
ICD10_LOOKUP_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
# NLM Clinical Tables also serves HCPCS (which is the superset that includes CPT-like codes
# for public lookup purposes). This replaces the old 4-row hardcoded cpt_master stub.
HCPCS_LOOKUP_URL = "https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search"


@st.cache_resource
def init_database():
    """Sets up the persistent feedback table used for closed-loop (835) active learning.
    Code validation no longer depends on a local table — see validate_icd10/validate_cpt."""
    try:
        engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS feedback_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_age INTEGER,
                        icd_10_group INTEGER,
                        cpt_code_group INTEGER,
                        provider_id INTEGER,
                        billed_amount FLOAT,
                        prior_auth_flag INTEGER,
                        coding_error_flag INTEGER,
                        denied INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.commit()
        return engine
    except Exception as e:
        st.error(f"Database Initialization Error: {e}")
        return None


db_engine = init_database()


def get_deterministic_group(code_str: str, max_buckets: int) -> int:
    """Hashes a code into a bucket so the (synthetic) ML model has a fixed-width feature,
    without needing a real code -> outcome mapping."""
    hex_dig = hashlib.sha256(code_str.strip().upper().encode("utf-8")).hexdigest()
    return int(hex_dig, 16) % max_buckets


@st.cache_data(ttl=3600, show_spinner=False)
def validate_icd10(code: str) -> tuple[bool, str]:
    """Validates against the real NLM Clinical Tables ICD-10-CM index."""
    if not code:
        return False, ""
    try:
        resp = requests.get(ICD10_LOOKUP_URL, params={"sf": "code,name", "terms": code}, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        matches = data[3] if len(data) > 3 else []
        for match in matches:
            if match[0].upper() == code.upper():
                return True, match[1]
    except requests.exceptions.RequestException:
        pass
    return False, ""


@st.cache_data(ttl=3600, show_spinner=False)
def validate_cpt_hcpcs(code: str) -> tuple[bool, str]:
    """Three-tier validation:
    1) Live NLM HCPCS API — covers Category I / HCPCS Level II codes.
    2) Local cpt_cat23_existence table — Category II (F-suffix) / III (T-suffix)
       codes parsed from a manually downloaded AMA PDF via cpt_code_existence_loader.py.
       Only confirms EXISTENCE, not description (AMA descriptor text is copyrighted
       and intentionally not stored/displayed here).
    3) Small local reference set — last-resort fallback for a few common demo codes
       if neither of the above is available (e.g. offline, or table not yet loaded).
    """
    if not code:
        return False, ""

    try:
        resp = requests.get(HCPCS_LOOKUP_URL, params={"sf": "code,short_desc", "terms": code}, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        matches = data[3] if len(data) > 3 else []
        for match in matches:
            if match[0].upper() == code.upper():
                return True, match[1]
    except requests.exceptions.RequestException:
        pass

    if db_engine is not None:
        try:
            with db_engine.connect() as conn:
                res = conn.execute(
                    text("SELECT category, loaded_at FROM cpt_cat23_existence WHERE code = :c"),
                    {"c": code.upper()},
                ).fetchone()
            if res:
                category, loaded_at = res
                label = "Category II — quality measure" if category == "II" else "Category III — emerging technology"
                return True, f"Valid {label} code (existence-verified from AMA data, loaded {loaded_at[:10]}). " \
                              f"See AMA CPT documentation for the full descriptor."
        except Exception:
            pass  # table may not exist yet if the loader script hasn't been run

    # Offline / supplemental reference set. Two reasons a code lands here instead of
    # the live API:
    #  1) the live NLM lookup is unreachable, or
    #  2) the code is a Category II (quality-measurement, suffix "F") or Category III
    #     (emerging-technology, suffix "T") CPT code. Full CPT — including Cat II/III —
    #     is AMA copyrighted content and is NOT covered by NLM's free public HCPCS API.
    #     There is no complete free public source for these; a production system would
    #     need a licensed AMA CPT data feed. This local set is a labeled stand-in so the
    #     app doesn't falsely flag well-known example codes as invalid.
    fallback = {
        "93000": "Electrocardiogram, routine ECG with at least 12 leads; complete",
        "93001": "Electrocardiogram, routine ECG with at least 12 leads; tracing only",
        "99213": "Office/outpatient visit, established patient, low-to-moderate complexity",
        "99214": "Office/outpatient visit, established patient, moderate-to-high complexity",
        "27245": "Treatment of intertrochanteric, peritrochanteric, or subtrochanteric "
        "femoral fracture; with intramedullary implant",
        "2022F": "Dilated macular exam performed and documented (diabetes care) "
        "— Category II quality-measurement code",
        "0795T": "Non-invasive digital tracking of retinal blood flow using specialized "
        "software — Category III emerging-technology code",
    }
    if code.upper() in fallback:
        suffix = code.upper()[-1]
        tag = " (Category II — quality measure)" if suffix == "F" else \
              " (Category III — emerging technology)" if suffix == "T" else \
              " (offline reference set)"
        return True, fallback[code.upper()] + tag
    return False, ""


# --- Session State ---
for key, default in {
    "ehr_connection_status": "Disconnected",
    "connected_ehr_vendor": None,
    "access_token": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def train_denial_model():
    """Trains on synthetic data with a hand-designed risk rule, seeded, plus any real
    closed-loop feedback logged via the 835 tab. This is a prototype pipeline, not a
    production model — see the notice banner at the top of the app."""
    np.random.seed(42)
    n_samples = 6000

    df = pd.DataFrame(
        {
            "patient_age": np.random.randint(18, 90, size=n_samples),
            "icd_10_group": np.random.randint(0, 10, size=n_samples),
            "cpt_code_group": np.random.randint(0, 15, size=n_samples),
            "provider_id": np.random.randint(100, 150, size=n_samples),
            "billed_amount": np.random.uniform(150.0, 18000.0, size=n_samples),
            "prior_auth_flag": np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75]),
            "coding_error_flag": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        }
    )

    risk_score = (
        (df["billed_amount"] > 6000).astype(int) * 2
        + (df["prior_auth_flag"] == 0).astype(int) * 4
        + (df["coding_error_flag"] == 1).astype(int) * 3
        + np.random.normal(0, 0.8, size=n_samples)
        > 3.5
    ).astype(int)
    df["denied"] = risk_score

    if db_engine is not None:
        try:
            feedback_df = pd.read_sql("SELECT * FROM feedback_memory", con=db_engine)
            if not feedback_df.empty:
                feedback_df = feedback_df.drop(columns=["id", "created_at"], errors="ignore")
                df = pd.concat([df, feedback_df], ignore_index=True)
        except Exception:
            pass

    feature_cols = [
        "patient_age", "icd_10_group", "cpt_code_group", "provider_id",
        "billed_amount", "prior_auth_flag", "coding_error_flag",
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X, y = df[feature_cols], df["denied"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) else 0.5

    return model, X_train, optimal_threshold


model, X_train_global, optimal_threshold = train_denial_model()


def get_payer_knowledge_graph_mandates(payer_name: str) -> dict:
    knowledge_base = {
        "BlueCross BlueShield (BCBS)": {
            "policy_ref": "BCBS Medical Policy Manual Section 7.01.42",
            "lcd_ncd": "LCD L38952 / NCD 220.6.1",
            "contractual_mandate": (
                "Requires documentation establishing direct medical necessity via "
                "peer-reviewed diagnostic criteria and prior clinical failure notes "
                "when billed amounts exceed baseline thresholds."
            ),
        },
        "UnitedHealthcare (UHC)": {
            "policy_ref": "UHC Commercial Medical Policy Guideline 2026.R4",
            "lcd_ncd": "LCD L34508 / NCD 190.2",
            "contractual_mandate": (
                "Mandates explicit alignment with UHC protocol guidelines, proof of "
                "step-therapy completion, and attending physician attestation for "
                "specialized CPT codes."
            ),
        },
        "Aetna Commercial": {
            "policy_ref": "Aetna Clinical Policy Bulletin (CPB) #0842",
            "lcd_ncd": "LCD L36281 / NCD 20.32",
            "contractual_mandate": (
                "Stipulates full disclosure of concurrent treatment plans, objective "
                "baseline functional scoring, and strict adherence to NCCI procedure edits."
            ),
        },
        "Medicare Part A/B (Novitas/Palmetto)": {
            "policy_ref": "CMS Medicare Program Integrity Manual Chapter 3",
            "lcd_ncd": "LCD L35036 & NCD 30.1",
            "contractual_mandate": (
                "Enforces statutory compliance with reasonable-and-necessary provisions "
                "under Title XVIII of the Social Security Act, requiring complete medical "
                "record corroboration."
            ),
        },
    }
    return knowledge_base.get(
        payer_name,
        {
            "policy_ref": "Standard Payer Guidelines Manual v4.2",
            "lcd_ncd": "General Administrative Review Standard",
            "contractual_mandate": "General contractual adherence to standard clinical necessity and timely filing limits.",
        },
    )


def categorize_denial_reason(prior_auth: str, coding_error: int, billed_amount: float) -> str:
    if prior_auth == "No":
        return "Missing Prior Authorization / Payer Pre-Certification Required"
    if coding_error == 1:
        return "Incorrect Medical Coding / NCCI Procedure Edit Mismatch"
    if billed_amount > 10000:
        return "Medically Unnecessary / High-Cost Threshold Review Required"
    return "General Administrative Discrepancy"


def simulate_fhir_ehr_intercept(ehr_system: str, mock_payload_id: int):
    np.random.seed(mock_payload_id)
    simulated_claims = {
        "patient_age": int(np.random.randint(22, 88)),
        "icd_10_group": int(np.random.randint(0, 10)),
        "cpt_code_group": int(np.random.randint(0, 15)),
        "provider_id": int(np.random.randint(100, 150)),
        "billed_amount": round(float(np.random.uniform(800.0, 15000.0)), 2),
        "prior_auth_flag": int(np.random.choice([0, 1], p=[0.3, 0.7])),
        "coding_error_flag": int(np.random.choice([0, 1], p=[0.8, 0.2])),
        "patient_id": f"MRN-{np.random.randint(100000, 999999)}",
        "encounter_date": datetime.today().strftime("%Y-%m-%d"),
    }
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "source_ehr": ehr_system,
        "entry": [
            {
                "resource": {
                    "resourceType": "Claim",
                    "id": simulated_claims["patient_id"],
                    "status": "active",
                    "use": "claim",
                    "patient": {"reference": f"Patient/{simulated_claims['patient_id']}"},
                    "created": simulated_claims["encounter_date"],
                    "provider": {"reference": f"Practitioner/{simulated_claims['provider_id']}"},
                    "total": {"value": simulated_claims["billed_amount"], "currency": "USD"},
                }
            }
        ],
    }
    return simulated_claims, fhir_bundle


# --- Sidebar: EHR Connection ---
st.sidebar.header("🔌 EHR Connection (OAuth 2.0 / FHIR R4)")
st.sidebar.caption(
    "Live connection requires a real EPIC_TOKEN_ENDPOINT and valid client credentials. "
    "Without them, this panel runs in simulation mode only."
)

selected_ehr_vendor = st.sidebar.selectbox(
    "Select Hospital Network / EHR Vendor",
    [
        "Epic Systems (Resolute PB)",
        "Oracle Health / Cerner Millennium",
        "Meditech Expanse API",
        "Athenahealth EHR Cloud",
    ],
)

auth_username = st.sidebar.text_input("API Client ID", value=os.getenv("EPIC_CLIENT_ID", ""))
auth_password = st.sidebar.text_input("API Client Secret", type="password")

col_btn1, col_btn2 = st.sidebar.columns(2)
auth_submitted = col_btn1.button("🔗 Authenticate")
disc_submitted = col_btn2.button("🔌 Disconnect")

if auth_submitted:
    if not auth_username or not auth_password:
        st.sidebar.error("⚠️ Please enter both client ID and client secret.")
    else:
        try:
            resp = requests.post(
                EPIC_TOKEN_ENDPOINT,
                data={
                    "grant_type": "client_credentials",
                    "client_id": auth_username,
                    "client_secret": auth_password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=6,
            )
            if resp.status_code == 200:
                st.session_state["access_token"] = resp.json().get("access_token")
                st.session_state["ehr_connection_status"] = "Connected"
                st.session_state["connected_ehr_vendor"] = selected_ehr_vendor
                st.sidebar.success("✅ Live token handshake successful.")
            else:
                st.sidebar.error(f"❌ Authentication failed (HTTP {resp.status_code}).")
        except requests.exceptions.RequestException:
            st.sidebar.warning(
                "⚠️ Could not reach live FHIR gateway — switch to 'Manual Sidebar Parameters' "
                "or 'Enterprise EHR API' simulation mode to continue exploring the app offline."
            )

if disc_submitted:
    st.session_state["ehr_connection_status"] = "Disconnected"
    st.session_state["connected_ehr_vendor"] = None
    st.session_state["access_token"] = None
    st.sidebar.warning("⚠️ Session terminated.")

status_ok = st.session_state["ehr_connection_status"] == "Connected"
badge_color = "#D1E7DD" if status_ok else "#F8D7DA"
badge_text_color = "#0F5132" if status_ok else "#842029"
badge_label = f"🟢 Connected ({st.session_state['connected_ehr_vendor']})" if status_ok else "🔴 Disconnected"
st.sidebar.markdown(
    f'<div style="background-color:{badge_color};color:{badge_text_color};padding:10px;'
    f'border-radius:5px;font-weight:600;text-align:center;margin-top:5px;">{badge_label}</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.header("🔌 Data Intake")
intake_mode = st.sidebar.radio(
    "Select Data Source Channel",
    ["Enterprise EHR API (Simulated FHIR)", "Manual Sidebar Parameters", "Batch CSV Upload"],
)

st.sidebar.markdown("---")
st.sidebar.header("🏢 Payer Knowledge Mapping")
selected_payer_target = st.sidebar.selectbox(
    "Target Insurance Payer",
    [
        "BlueCross BlueShield (BCBS)",
        "UnitedHealthcare (UHC)",
        "Aetna Commercial",
        "Medicare Part A/B (Novitas/Palmetto)",
    ],
)

# --- Defaults ---
patient_age, billed_amount = 45, 4500.0
prior_auth, prior_auth_numeric = "Yes", 1
coding_error_input, coding_numeric = "None", 0
icd_code_input, icd_description = "R50.9", "Fever, unspecified"
cpt_code_input, cpt_description = "93000", "Electrocardiogram, routine ECG with at least 12 leads"
icd_group = get_deterministic_group(icd_code_input, 10)
cpt_group = get_deterministic_group(cpt_code_input, 15)
fhir_json_payload = {}
icd_valid, cpt_valid = True, True

if intake_mode == "Enterprise EHR API (Simulated FHIR)":
    st.sidebar.subheader("FHIR Gateway Settings")
    is_connected = st.session_state.get("ehr_connection_status") == "Connected"
    if is_connected:
        st.sidebar.success(f"🟢 Linked to {st.session_state.get('connected_ehr_vendor')}")
        selected_ehr = st.session_state.get("connected_ehr_vendor")
    else:
        st.sidebar.info("Running in offline simulation (no live EHR session).")
        selected_ehr = st.sidebar.selectbox(
            "Simulate EHR Instance", ["Epic Systems (Resolute PB)", "Meditech Expanse API"]
        )

    payload_id_trigger = st.sidebar.number_input(
        "Simulate Inbound Claim ID", min_value=1, max_value=9999, value=1042
    )
    raw_claim_data, fhir_json_payload = simulate_fhir_ehr_intercept(selected_ehr, payload_id_trigger)
    patient_age = raw_claim_data["patient_age"]
    billed_amount = raw_claim_data["billed_amount"]
    prior_auth_numeric = raw_claim_data["prior_auth_flag"]
    prior_auth = "No" if prior_auth_numeric == 0 else "Yes"
    coding_numeric = raw_claim_data["coding_error_flag"]
    coding_error_input = "Flagged Mismatch" if coding_numeric == 1 else "None"
    icd_group, cpt_group = raw_claim_data["icd_10_group"], raw_claim_data["cpt_code_group"]
    icd_code_input, icd_description = "R07.9", "Chest pain, unspecified"
    cpt_code_input, cpt_description = "93000", "Electrocardiogram, routine ECG with at least 12 leads"

elif intake_mode == "Manual Sidebar Parameters":
    st.sidebar.header("📋 Manual Claim Parameters")
    patient_age = st.sidebar.slider("Patient Age", 18, 90, 58)
    billed_amount = st.sidebar.number_input("Billed Amount ($)", min_value=50.0, max_value=50000.0, value=4500.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Diagnostic Coding (ICD-10-CM)")
    icd_code_input = st.sidebar.text_input("ICD-10 Diagnostic Code", value="R50.9").strip().upper()
    icd_valid, icd_description = validate_icd10(icd_code_input)
    if not icd_valid:
        icd_description = f"[INVALID / UNVERIFIED ICD-10 CODE: {icd_code_input}]"
        st.sidebar.error(f"❌ Invalid or Unrecognized ICD-10 Code: `{icd_code_input}`")
    st.sidebar.markdown(f"**Description:** *{icd_description}*")
    icd_group = get_deterministic_group(icd_code_input, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💉 Procedure Coding (CPT / HCPCS)")
    st.sidebar.caption(
        "Category I codes validate live via NLM HCPCS lookup. Category II (suffix F) "
        "and Category III (suffix T) codes use a small local reference set, since full "
        "CPT is AMA-licensed content not available via free public API."
    )
    cpt_code_input = st.sidebar.text_input("CPT / HCPCS Procedure Code", value="93000").strip().upper()
    cpt_valid, cpt_description = validate_cpt_hcpcs(cpt_code_input)
    if not cpt_valid:
        cpt_description = f"[INVALID / UNVERIFIED CPT CODE: {cpt_code_input}]"
        st.sidebar.error(f"❌ Invalid or Unrecognized CPT Code: `{cpt_code_input}`")
    st.sidebar.markdown(f"**Description:** *{cpt_description}*")
    cpt_group = get_deterministic_group(cpt_code_input, 15)

    st.sidebar.markdown("---")
    prior_auth = st.sidebar.selectbox("Prior Authorization Obtained?", ["Yes", "No"])
    coding_error_input = st.sidebar.selectbox("Potential Coding Flag?", ["None", "Flagged Mismatch"])
    coding_numeric = 1 if coding_error_input == "Flagged Mismatch" else 0
    prior_auth_numeric = 0 if prior_auth == "No" else 1

input_data = pd.DataFrame(
    {
        "patient_age": [patient_age],
        "icd_10_group": [icd_group],
        "cpt_code_group": [cpt_group],
        "provider_id": [105],
        "billed_amount": [billed_amount],
        "prior_auth_flag": [prior_auth_numeric],
        "coding_error_flag": [coding_numeric],
    }
)

denial_probability = float(model.predict_proba(input_data)[:, 1][0])
primary_risk_factor = categorize_denial_reason(prior_auth, coding_numeric, billed_amount)
payer_mandates = get_payer_knowledge_graph_mandates(selected_payer_target)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⚡ Single Claim Predictor & Appeal",
        "🔌 FHIR Intercept Inspector",
        "📊 Batch Claims Processing",
        "🔍 Model Insights & Explainability",
        "🔄 Closed-Loop Feedback (835 Active Learning)",
    ]
)

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📊 Risk Evaluation Dashboard")
        if not icd_valid or not cpt_valid:
            st.error(
                "🚨 **Compliance Alert:** Claim contains invalid/unverified diagnostic or "
                "procedural codes. Correction required before submission."
            )
            denial_probability = 0.99

        st.metric(
            label="Predicted Denial Probability",
            value=f"{denial_probability * 100:.1f}%",
            delta=f"Threshold: {optimal_threshold*100:.1f}%",
            delta_color="inverse",
        )

        if denial_probability >= optimal_threshold:
            st.error("⚠️ **High Risk:** Claim likely to trigger denial. Pre-submission review recommended.")
        else:
            st.success("✅ **Low Risk:** Claim meets baseline acceptance criteria (per prototype model).")

        st.markdown("**Primary Root-Cause Driver:**")
        st.info(primary_risk_factor)

        st.markdown("---")
        st.subheader("🌐 Payer Knowledge Mapping")
        st.markdown(f"**Target Payer:** `{selected_payer_target}`")
        st.markdown(f"**Policy Reference:** `{payer_mandates['policy_ref']}`")
        st.markdown(f"**Applicable LCD/NCD:** `{payer_mandates['lcd_ncd']}`")

    with col2:
        st.subheader("✉️ Draft Clinical Appeal Letter")
        appeal_letter = f"""[Date: {pd.Timestamp.today().strftime('%B %d, %Y')}]

To: Utilization Review & Claims Appeals Department
{selected_payer_target} - Medical Policy Administration

Subject: Formal Appeal for Reconsideration & Payment Authorization
------------------------------------------------------------------
Patient Age: {patient_age} | Diagnostic Code: {icd_code_input} ({icd_description})
Procedure Code: {cpt_code_input} ({cpt_description}) | Total Billed: ${billed_amount:.2f}

Governing Payer Policy Reference: {payer_mandates['policy_ref']}
Applicable Coverage Determinations: {payer_mandates['lcd_ncd']}

Dear Medical Review Board,

We are writing to formally appeal the rejection status or projected denial risk
associated with the referenced medical claim. Under {payer_mandates['policy_ref']}
({payer_mandates['lcd_ncd']}), {payer_mandates['contractual_mandate']}

Regarding the identified operational flag for '{primary_risk_factor}', provider
documentation supports compliance with standard clinical guidelines and payer
contract specifications.

We request an expedited review of the stated billing balance.

Sincerely,
Clinical Documentation & Revenue Cycle Management Team
PrimEra Medical Technologies

[DRAFT — auto-generated; requires clinical and compliance review before sending.]
"""
        st.text_area("Generated Draft Letter", value=appeal_letter, height=350)
        st.download_button(
            "📥 Download Draft Appeal Letter (.txt)",
            data=appeal_letter,
            file_name=f"PrimEra_Appeal_Draft_{int(billed_amount)}.txt",
            mime="text/plain",
        )

with tab2:
    st.subheader("🔌 FHIR API Interception Monitor (Simulated)")
    if st.session_state.get("ehr_connection_status") == "Connected":
        selected_ehr = st.session_state.get("connected_ehr_vendor", "Epic Systems")
    else:
        selected_ehr = "Epic Systems (Resolute PB) — simulated, no live session"

    col_f1, col_f2 = st.columns(2)
    col_f1.markdown(f"**Endpoint:** `{selected_ehr}`")
    col_f1.markdown(f"**Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    col_f1.markdown(
        f"**Calculated Risk Level:** `{'HIGH RISK' if denial_probability >= optimal_threshold else 'CLEAN PASS'}`"
    )
    col_f2.metric("Payload Validation", "Simulated bundle" if not status_ok else "Live session active")

    st.markdown("#### Sample FHIR R4 Bundle Payload")
    if intake_mode == "Enterprise EHR API (Simulated FHIR)" and fhir_json_payload:
        st.json(fhir_json_payload)
    else:
        _, sample_bundle = simulate_fhir_ehr_intercept("Epic Systems (Resolute PB)", 1042)
        st.json(sample_bundle)

with tab3:
    st.subheader("📁 Batch Claims Processing & Bulk Scoring")
    uploaded_file = st.file_uploader("Upload Claims Dataset (.csv)", type=["csv"])
    required_cols = [
        "patient_age", "icd_10_group", "cpt_code_group", "provider_id",
        "billed_amount", "prior_auth_flag", "coding_error_flag",
    ]
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", batch_df.head(3))
        if st.button("Run Bulk Scoring"):
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            if missing_cols:
                st.warning(f"Missing columns, cannot score: {missing_cols}")
            else:
                probs = model.predict_proba(batch_df[required_cols])[:, 1]
                batch_df["Predicted_Denial_Probability"] = probs
                batch_df["Action_Plan"] = batch_df["Predicted_Denial_Probability"].apply(
                    lambda x: "Hold & Attach Clinical Notes" if x >= optimal_threshold else "Auto-Submit to Clearinghouse"
                )
                st.success("Batch scoring complete.")
                st.dataframe(batch_df)
                st.download_button(
                    "Download Scored Batch Report (.csv)",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="PrimEra_Batch_Scored_Claims.csv",
                    mime="text/csv",
                )
    else:
        if st.button("Generate Sample CSV Template"):
            mock_template = pd.DataFrame(
                {
                    "patient_age": [45, 62, 31, 78],
                    "icd_10_group": [2, 5, 1, 8],
                    "cpt_code_group": [3, 10, 4, 12],
                    "provider_id": [101, 102, 103, 104],
                    "billed_amount": [1200.50, 8400.00, 450.00, 14200.00],
                    "prior_auth_flag": [1, 0, 1, 0],
                    "coding_error_flag": [0, 0, 1, 0],
                }
            )
            st.dataframe(mock_template)
            st.download_button(
                "Download Sample CSV",
                data=mock_template.to_csv(index=False).encode("utf-8"),
                file_name="sample_claims_template.csv",
                mime="text/csv",
            )

with tab4:
    st.subheader("🔍 Explainable AI (SHAP) & Model Diagnostics")
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("#### Global Feature Importance (XGBoost gain)")
        fig, ax = plt.subplots(figsize=(6, 4))
        xgb.plot_importance(model, ax=ax, color="#1E3A8A", grid=False, max_num_features=6)
        st.pyplot(fig)

        st.markdown("#### SHAP Summary (sample of training data)")
        try:
            explainer = shap.TreeExplainer(model)
            sample = X_train_global.sample(min(200, len(X_train_global)), random_state=42)
            shap_values = explainer.shap_values(sample)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, sample, show=False, plot_size=None)
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

    with col_x2:
        st.markdown("#### Clinical Audit Notes")
        st.markdown(
            """
            * **Prior Auth Status:** highest-weight feature for denial likelihood in this model.
            * **Billed Amount Threshold:** high-cost claims trigger the synthetic secondary-review rule.
            * **Coding Discrepancies:** mismatched ICD-10/CPT pairs are modeled as a denial driver.

            *Note: because the model is trained on synthetic data with a hand-designed risk
            rule (see `train_denial_model`), these feature importances reflect that rule,
            not empirically observed payer behavior, until real 835 feedback (tab 5)
            accumulates.*
            """
        )

with tab5:
    st.subheader("🔄 Closed-Loop Feedback & Active Learning (Persistent DB)")
    st.markdown(
        "Logs actual 835 Electronic Remittance Advice outcomes to a persistent SQL table. "
        "Once enough real outcomes accumulate, the model retrains on real + synthetic data blended."
    )

    col_fb1, col_fb2 = st.columns(2, gap="large")
    with col_fb1:
        st.markdown("#### 📥 Log Inbound 835 Remittance Outcome")
        era_status = st.selectbox(
            "Actual Payer Adjudication Result",
            ["Paid / Accepted (Ground Truth: 0)", "Denied / Rejected (Ground Truth: 1)"],
        )
        fb_denied_val = 1 if "Denied" in era_status else 0

        if st.button("Log 835 Feedback to Database"):
            row_data = {
                "patient_age": patient_age,
                "icd_10_group": icd_group,
                "cpt_code_group": cpt_group,
                "provider_id": 105,
                "billed_amount": billed_amount,
                "prior_auth_flag": prior_auth_numeric,
                "coding_error_flag": coding_numeric,
                "denied": fb_denied_val,
            }
            try:
                if db_engine is not None:
                    pd.DataFrame([row_data]).to_sql(
                        "feedback_memory", con=db_engine, if_exists="append", index=False, method="multi"
                    )
                    train_denial_model.clear()
                    st.success("✅ Feedback logged. Model will retrain on next run.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to log feedback: {e}")

    with col_fb2:
        st.markdown("#### 📈 Active Learning Stats")
        try:
            total_logs = pd.read_sql("SELECT COUNT(*) FROM feedback_memory", con=db_engine).iloc[0, 0]
        except Exception:
            total_logs = 0
        st.metric("Total 835 Outcomes Logged", total_logs)
        st.markdown(
            "* Feedback is stored persistently in SQL.\n"
            "* Designed to accept real webhook-fed 835 outcomes in a production deployment."
        )