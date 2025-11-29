import streamlit as st
import pandas as pd
import pdfplumber
import ofxparse
import io
import re

st.set_page_config(page_title="Bank Statement Classifier – AI", layout="wide")
st.title("🏦 Bank Statement Classifier – Universal AI Parser")

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
# 🔹 CSV / XLSX PARSER
###########################################
def parse_excel_csv(file):
    try:
        return pd.read_csv(file)
    except:
        file.seek(0)
        return pd.read_excel(file)

###########################################
# 🔹 PDF PARSER (TEXT ONLY)
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
# 🔹 OFX/QFX/QBO PARSER
###########################################
def parse_ofx(file):
    data = ofxparse.OfxParser.parse(file)
    rows = []
    for tx in data.account.statement.transactions:
        rows.append({
            "Date": tx.date.strftime("%Y-%m-%d"),
            "Description": tx.memo,
            "Amount": tx.amount,
            "Balance": None,
        })
    return pd.DataFrame(rows)

###########################################
# 🔹 MASTER PARSER (AUTO)
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
        return parse_ofx(uploaded)

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
