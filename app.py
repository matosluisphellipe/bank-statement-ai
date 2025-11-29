import streamlit as st
import pandas as pd
import pdfplumber
import io
import re

st.set_page_config(page_title="Bank Statement Classifier – AI", layout="wide")
st.title("🏦 Bank Statement Classifier – Bookkepper")

uploaded = st.file_uploader(
    "Upload your bank statement",
    type=["txt", "csv", "xlsx", "pdf", "ofx", "qfx", "qbo"]
)

###########################################
# 🔹 TXT PARSER
###########################################
def parse_txt(text):
    pattern = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})$"
    )

    rows = []
    for line in text.splitlines():
        m = pattern.search(line.strip())
        if m:
            d, desc, amt, bal = m.groups()
            rows.append({
                "Date": d,
                "Description": desc,
                "Amount": float(amt.replace(",", "")),
                "Balance": float(bal.replace(",", "")),
            })
    return pd.DataFrame(rows)

###########################################
# 🔹 OFX / QFX / QBO (Manual Parser)
###########################################
def parse_ofx_text(text):
    rows = []

    # Extract each transaction block <STMTTRN> ... </STMTTRN>
    blocks = re.findall(r"<STMTTRN>(.*?)</STMTTRN>", text, flags=re.DOTALL)

    for block in blocks:
        date_match = re.search(r"<DTPOSTED>(.*?)<", block)
        memo_match = re.search(r"<MEMO>(.*?)<", block)
        amount_match = re.search(r"<TRNAMT>(.*?)<", block)

        if amount_match:
            amount = float(amount_match.group(1))
        else:
            amount = None

        if date_match:
            raw_date = date_match.group(1)
            # Formats like: 20241005, 20241005120000[-5:EST]
            clean_date = re.sub(r"[^0-9]", "", raw_date)[:8]
            date_fmt = f"{clean_date[4:6]}/{clean_date[6:8]}/{clean_date[:4]}"
        else:
            date_fmt = None

        memo = memo_match.group(1) if memo_match else None

        rows.append({
            "Date": date_fmt,
            "Description": memo,
            "Amount": amount,
            "Balance": None
        })

    return pd.DataFrame(rows)

###########################################
# 🔹 CSV / XLSX PARSER
###########################################
def parse_excel_csv(file):
    try:
        return pd.read_csv(file)
    except:
        file.seek(0)
        return pd.read_excel(file)

###########################################
# 🔹 PDF PARSER (TEXT-ONLY)
###########################################
def parse_pdf_text(file):
    rows = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            df = parse_txt(text)
            if not df.empty:
                rows.append(df)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

###########################################
# 🔹 MASTER PARSER
###########################################
def parse_file(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".txt"):
        text = uploaded.read().decode("utf-8", errors="ignore")
        return parse_txt(text)

    if name.endswith(".csv") or name.endswith(".xlsx"):
        return parse_excel_csv(uploaded)

    if name.endswith(".pdf"):
        return parse_pdf_text(uploaded)

    if name.endswith((".ofx", ".qfx", ".qbo")):
        text = uploaded.read().decode("utf-8", errors="ignore")
        return parse_ofx_text(text)

    raise ValueError("Unsupported format")

###########################################
# 🔹 PROCESSAMENTO
###########################################
if uploaded:
    try:
        df = parse_file(uploaded)

        if df.empty:
            st.warning("⚠️ File processed but no transactions were detected.")
        else:
            st.success("✅ File processed successfully!")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
