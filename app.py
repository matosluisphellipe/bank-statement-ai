import streamlit as st
import pandas as pd
import pdfplumber
import io

st.set_page_config(
    page_title="Bank Statement Classifier – AI",
    layout="wide"
)

# -------------------------------------------------------
# FUNÇÃO PARA PROCESSAR ARQUIVOS
# -------------------------------------------------------

def process_file(uploaded_file):
    filename = uploaded_file.name.lower()

    # ----------------------
    # Caso seja PDF
    # ----------------------
    if filename.endswith(".pdf"):
        text_data = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text_data.append(page.extract_text())

        # transforma cada linha em uma "linha de tabela"
        df = pd.DataFrame({"Description": text_data})
        return df

    # ----------------------
    # Caso seja XLSX
    # ----------------------
    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    # ----------------------
    # Caso seja CSV
    # Leitura robusta (NÃO quebra com tokenizing error!)
    # ----------------------
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, engine="python", sep=None)  # detecta delimitador
        except:
            try:
                df = pd.read_csv(uploaded_file, engine="python", sep=",", on_bad_lines="skip")
            except:
                df = pd.read_csv(
                    uploaded_file,
                    engine="python",
                    sep=",",
                    header=None,
                    names=["col1", "col2", "col3", "col4"],
                    on_bad_lines="skip"
                )

        df = df.dropna(axis=1, how="all")  # remove colunas vazias
        df = df.dropna(how="all")          # remove linhas vazias
        return df

    return None


# -------------------------------------------------------
# INTERFACE
# -------------------------------------------------------

st.title("📊 Bank Statement Classifier – AI")
st.write("Upload your bank statement in **PDF**, **CSV**, or **XLSX** format.")

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "csv", "xlsx"]
)

if uploaded_file:
    try:
        df = process_file(uploaded_file)
        st.success("File processed successfully!")

        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("Please upload a file to begin.")
