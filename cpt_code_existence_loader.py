"""
Local-file loader for CPT Category II / III code EXISTENCE validation only.

WHAT THIS DOES
---------------
Parses just the code numbers (e.g. 0559T, 1030T, 2022F) out of a manually-downloaded
AMA PDF and stores them in a local table for fast lookup. It does NOT store or
display AMA's copyrighted descriptor text — only whether a code currently exists
in the published Category II/III set.

WHY NO DESCRIPTIONS
---------------------
The PDF descriptor text is AMA copyrighted content. Storing/redistributing it
through this app's UI would go beyond what AMA's free-viewing PDF release permits.
Code numbers themselves (five-character identifiers) are treated separately —
this app only asserts "code exists / does not exist," not "here is the definition."
If a user needs the definition, point them to the AMA CPT resource directly.

HOW TO USE
-----------
1. In Chrome, search: AMA CPT Category III codes PDF  (or Category II)
2. Download the PDF to your machine, e.g. C:\\Users\\you\\Downloads\\cpt_cat3.pdf
3. Run:  python cpt_code_existence_loader.py "C:\\path\\to\\cpt_cat3.pdf" III
   (or "...cat2.pdf" II for Category II)
4. Re-run whenever AMA republishes (Category III: semi-annual; Category II: periodic)
   with a freshly downloaded PDF to keep the table current.
"""

import re
import sqlite3
import sys
from datetime import datetime, timezone

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Category III: four digits + T, OR newer alphanumeric form (one letter + 3 digits + T)
CAT_III_CODE_PATTERN = re.compile(r"\b([0-9]{4}T|[A-Z][0-9]{3}T)\b")
# Category II: four digits + F
CAT_II_CODE_PATTERN = re.compile(r"\b([0-9]{4}F)\b")

DB_PATH = "rcm_production_master.db"


def extract_codes_from_pdf(pdf_path: str, category: str) -> set:
    if pdfplumber is None:
        raise RuntimeError("Install pdfplumber first: pip install pdfplumber")

    pattern = CAT_III_CODE_PATTERN if category == "III" else CAT_II_CODE_PATTERN
    codes = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for match in pattern.finditer(text):
                codes.add(match.group(1).upper())

    return codes


def save_codes(codes: set, category: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cpt_cat23_existence (
            code TEXT PRIMARY KEY,
            category TEXT,
            loaded_at TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    # Clear old entries for this category before loading fresh ones, so codes
    # that were removed/archived in a newer PDF don't linger as false positives.
    conn.execute("DELETE FROM cpt_cat23_existence WHERE category = ?", (category,))
    for code in codes:
        conn.execute(
            "INSERT OR REPLACE INTO cpt_cat23_existence (code, category, loaded_at) VALUES (?, ?, ?)",
            (code, category, now),
        )
    conn.commit()
    conn.close()


def code_exists(code: str, db_path: str = DB_PATH):
    """Returns (exists: bool, category: str|None, loaded_at: str|None)."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT category, loaded_at FROM cpt_cat23_existence WHERE code = ?",
        (code.upper(),),
    ).fetchone()
    conn.close()
    if row:
        return True, row[0], row[1]
    return False, None, None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("II", "III"):
        print('Usage: python cpt_code_existence_loader.py "<path_to_pdf>" <II|III>')
        sys.exit(1)

    pdf_path, category = sys.argv[1], sys.argv[2]
    print(f"Parsing Category {category} codes from: {pdf_path}")
    codes = extract_codes_from_pdf(pdf_path, category)
    print(f"Found {len(codes)} Category {category} codes.")

    if len(codes) < 5:
        print(
            "WARNING: very few codes found — the PDF layout may not match the "
            "expected pattern. Inspect the extracted text before trusting this run."
        )
    else:
        save_codes(codes, category)
        print(f"Saved to {DB_PATH} (table: cpt_cat23_existence).")