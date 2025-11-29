import streamlit as st
import pandas as pd
import pdfplumber

st.set_page_config(page_title="Bank Statement Classifier – AI", layout="wide")

st.title("📊 Bank Statement Classifier – AI")
st.write("Upload your bank statement in **PDF, CSV, or XLSX** format.")

def process_csv(file):
    # Load even if the CSV has broken or uneven rows
    df = pd.read_csv(file, engine="python", on_bad_lines="skip")
    return df

def process_excel(file):
    df = pd.read_excel(file)
    return df

def process_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    df = pd.DataFrame({"PDF_Text": [text]})
    return df

uploaded_file = st.file_uploader(
    "Upload your file", 
    type=["pdf", "csv", "xlsx"],
    help="Supported formats: PDF, CSV, XLSX"
)

if uploaded_file:
    try:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "csv":
            df = process_csv(uploaded_file)
        elif file_type == "xlsx":
            df = process_excel(uploaded_file)
        else:
            df = process_pdf(uploaded_file)

        st.success("File processed successfully!")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
