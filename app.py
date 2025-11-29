import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Bank Statement Classifier – AI", layout="wide")

st.title("🏦 Bank Statement Classifier – AI")
st.write("Upload your bank statement in **TXT**, **PDF**, **CSV**, or **XLSX** format.")

uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv", "xlsx", "pdf"])

def parse_txt_statement(text):
    """
    Extrai transações do arquivo TXT exatamente no formato do seu extrato.
    """
    lines = text.splitlines()

    data = []
    pattern = re.compile(r"^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})$")

    for line in lines:
        match = pattern.search(line.strip())
        if match:
            date, description, amount, balance = match.groups()
            data.append({
                "Date": date,
                "Description": description.strip(),
                "Amount": float(amount.replace(",", "")),
                "Balance After": float(balance.replace(",", "")),
            })

    return pd.DataFrame(data)


if uploaded_file is not None:
    file_type = uploaded_file.name.lower()

    try:
        if file_type.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            df = parse_txt_statement(text)

        elif file_type.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_type.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("PDF ainda não está habilitado nesta versão.")
            st.stop()

        if df.empty:
            st.warning("⚠️ The file was processed, but no transactions were found.")
        else:
            st.success("✅ File processed successfully!")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
