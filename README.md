# PrimEra RCM — Claims Denial-Risk Prototype

A Streamlit application prototyping an end-to-end revenue cycle management (RCM) workflow: diagnostic/procedure code validation, ML-based claims denial-risk scoring, payer policy mapping, and automated appeal-letter drafting.

## What this project demonstrates

- **Real-data code validation** — ICD-10-CM and CPT/HCPCS Category I codes are validated live against the NLM Clinical Tables API, not a static list.
- **Layered validation strategy for edge cases** — CPT Category II/III codes (which have no free public API) are validated for existence via a locally maintained reference table, sourced from AMA's periodic public releases. See [Data Sourcing & Licensing](#data-sourcing--licensing) below for why this is architected the way it is.
- **A full ML pipeline** — feature engineering, XGBoost classification, precision-recall threshold tuning, and SHAP-based explainability.
- **A closed-loop feedback mechanism** — a persistent store for real adjudication outcomes (835 remittance data), designed so the model can eventually retrain on ground truth instead of synthetic assumptions.
- **Simulated FHIR/EHR integration** — an OAuth2 handshake pattern and FHIR R4 bundle structure, so the intake layer is architected against a real interoperability standard even where the live connection is mocked.

## Architecture

```
Data Intake (EHR/FHIR sim | Manual | Batch CSV)
        │
        ▼
Code Validation Layer
   ├── ICD-10-CM  → live NLM API
   └── CPT/HCPCS  → live NLM API → local Cat II/III existence table → fallback set
        │
        ▼
Feature Engineering (deterministic hashing → fixed-width buckets)
        │
        ▼
XGBoost Denial-Risk Classifier (+ SHAP explainability)
        │
        ▼
Payer Policy Mapping → Appeal Letter Generation
        │
        ▼
835 Outcome Logging → Model Retraining Loop
```

## Data Sourcing & Licensing

This project makes a deliberate, documented distinction between code sets with different licensing terms:

| Code Set | Source | What's Stored |
|---|---|---|
| ICD-10-CM | NLM Clinical Tables API (free, public) | Full code + description |
| CPT Category I / HCPCS Level II | NLM Clinical Tables API (free, public) | Full code + description |
| CPT Category II (quality measures) | AMA public release (parsed locally via `cpt_code_existence_loader.py`) | **Code existence only** |
| CPT Category III (emerging tech) | AMA public release (parsed locally) | **Code existence only** |
| Full CPT Category I (AMA-licensed) | Not covered | — |

**Why Category II/III store existence only, not descriptions:** AMA publishes these two code sets publicly to enable early implementation, but the descriptor text itself remains AMA copyrighted content. This project validates that a submitted code is currently active without redistributing AMA's proprietary descriptions — the same boundary any RCM vendor operates under unless they hold a commercial CPT data license. Run `cpt_code_existence_loader.py` against a freshly downloaded AMA PDF to refresh the local table (Category III updates semi-annually).

## Model Notes

The denial-risk classifier is trained on **synthetic claims data** with a hand-specified risk heuristic (prior authorization status, billed amount thresholds, coding-error flags), not on real historical adjudication outcomes. This is intentional: it lets the full pipeline — features, training, threshold selection, explainability, feedback ingestion — be demonstrated end-to-end without requiring access to production claims data.

The `feedback_memory` table and Tab 5 (Closed-Loop Feedback) exist specifically to close this gap: as real 835 remittance outcomes are logged, they're blended into training data on each retrain cycle. In a production deployment, the synthetic baseline would be phased out once sufficient real-outcome volume accumulates.

**Practical implication:** predicted probabilities reflect the modeled heuristic, not empirically observed payer behavior, until real feedback volume is substantial.

## Setup

```bash
pip install -r requirements.txt
streamlit run primera_rcm_app.py
```

Optional — refresh CPT Category II/III existence data:
```bash
python cpt_code_existence_loader.py "path/to/ama-category3.pdf" III
python cpt_code_existence_loader.py "path/to/ama-category2.pdf" II
```

## Tech Stack

Python · Streamlit · XGBoost · SHAP · scikit-learn · SQLAlchemy · pandas/NumPy · pdfplumber

## Known Limitations

- EHR OAuth2 connection requires a real `EPIC_TOKEN_ENDPOINT` and credentials; without them the app runs in simulation mode.
- Denial-risk probabilities are illustrative pending real 835 feedback volume (see Model Notes).
- Full CPT Category I remains AMA-licensed content and is out of scope for this prototype's free-data-source approach.
- Category II/III existence data requires periodic manual refresh (no free live API exists for these code sets).

## License / Attribution

This project is a technical prototype. CPT is a registered trademark of the American Medical Association; ICD-10-CM is maintained by CMS/NCHS. This project does not redistribute AMA's copyrighted CPT descriptor text.
