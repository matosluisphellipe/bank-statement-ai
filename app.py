import io
import re

import pandas as pd
import pdfplumber
import streamlit as st


st.set_page_config(
    page_title="Bank Statement Parser & Viewer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------
def parse_txt(text: str) -> pd.DataFrame:
    pattern = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})\s+(-?\d{1,3}(?:,\d{3})*\.\d{2})$"
    )

    rows = []
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if match:
            date, desc, amt, bal = match.groups()
            rows.append(
                {
                    "Date": date,
                    "Description": desc,
                    "Amount": float(amt.replace(",", "")),
                    "Balance": float(bal.replace(",", "")),
                }
            )
    return pd.DataFrame(rows)


def parse_ofx_text(text: str) -> pd.DataFrame:
    rows = []
    blocks = re.findall(r"<STMTTRN>(.*?)</STMTTRN>", text, flags=re.DOTALL)

    for block in blocks:
        date_match = re.search(r"<DTPOSTED>(.*?)<", block)
        memo_match = re.search(r"<MEMO>(.*?)<", block)
        amount_match = re.search(r"<TRNAMT>(.*?)<", block)

        amount = float(amount_match.group(1)) if amount_match else None
        date_fmt = None

        if date_match:
            raw_date = date_match.group(1)
            clean_date = re.sub(r"[^0-9]", "", raw_date)[:8]
            if len(clean_date) == 8:
                date_fmt = f"{clean_date[4:6]}/{clean_date[6:8]}/{clean_date[:4]}"

        rows.append(
            {
                "Date": date_fmt,
                "Description": memo_match.group(1) if memo_match else None,
                "Amount": amount,
                "Balance": None,
            }
        )

    return pd.DataFrame(rows)


def parse_excel_csv(buffer: io.BytesIO) -> pd.DataFrame:
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_excel(buffer)


def parse_pdf_text(buffer: io.BytesIO) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(buffer) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            df_page = parse_txt(text)
            if not df_page.empty:
                rows.append(df_page)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def parse_file(uploaded_file) -> pd.DataFrame:
    raw_data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return parse_txt(raw_data.decode("utf-8", errors="ignore"))

    if name.endswith((".csv", ".xlsx")):
        return parse_excel_csv(io.BytesIO(raw_data))

    if name.endswith(".pdf"):
        return parse_pdf_text(io.BytesIO(raw_data))

    if name.endswith((".ofx", ".qfx", ".qbo")):
        text = raw_data.decode("utf-8", errors="ignore")
        return parse_ofx_text(text)

    raise ValueError("Unsupported format")


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "Amount" in normalized.columns:
        normalized["Amount"] = pd.to_numeric(normalized["Amount"], errors="coerce")
    return normalized.dropna(subset=["Amount"]).reset_index(drop=True)


def format_currency(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"${value:,.2f}"


def calculate_summary(df: pd.DataFrame) -> dict:
    total_in = df.loc[df["Amount"] > 0, "Amount"].sum()
    total_out = df.loc[df["Amount"] < 0, "Amount"].sum()
    return {
        "entries": total_in,
        "exits": total_out,
        "count": len(df),
    }


def render_metrics(summary: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de entradas", format_currency(summary["entries"]))
    col2.metric("Total de saídas", format_currency(summary["exits"]))
    col3.metric("Número de transações", f"{summary['count']:,}")


def render_header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.markdown(f"<p style='color:#6c757d;font-size:16px;'>{subtitle}</p>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Navigation state
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"


def go_to_details():
    st.session_state.page = "details"
    st.rerun()


def go_to_main():
    st.session_state.page = "main"
    st.rerun()


page = st.session_state.page


# ------------------------------------------------------------
# Page: Summary (upload + metrics)
# ------------------------------------------------------------
if page == "main":
    render_header(
        "Bank Statement Parser",
        "Envie seu extrato e visualize um resumo claro e profissional.",
    )

    with st.container():
        st.subheader("Upload do extrato")
        uploaded = st.file_uploader(
            "Envie arquivos PDF, CSV, XLSX, TXT, OFX, QFX ou QBO",
            type=["pdf", "csv", "xlsx", "txt", "ofx", "qfx", "qbo"],
            help="Aceitamos arquivos comuns de extratos bancários.",
        )

    if uploaded:
        try:
            df = normalize_transactions(parse_file(uploaded))
            if df.empty:
                st.warning("⚠️ Arquivo processado, mas nenhuma transação foi identificada.")
            else:
                df = df.dropna(axis=1, how="all")
                st.session_state.transactions = df
                summary = calculate_summary(df)

                st.success("✅ Arquivo processado com sucesso!")
                st.markdown("---")
                st.subheader("Resumo do extrato")
                render_metrics(summary)

                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                st.button("📄 Ver detalhes do extrato", type="primary", on_click=go_to_details)

        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Erro ao processar o arquivo: {exc}")
    else:
        st.info("Faça o upload do extrato para ver o resumo.")


# ------------------------------------------------------------
# Page: Details (full table)
# ------------------------------------------------------------
if page == "details":
    render_header(
        "📄 Detalhes do extrato",
        "Visualize e filtre todas as transações processadas.",
    )

    df = st.session_state.get("transactions")
    if df is None or df.empty:
        st.info("Nenhum extrato carregado ainda. Volte para a página de Resumo e envie um arquivo.")
        st.button("⬅️ Voltar", on_click=go_to_main)
    else:
        df = normalize_transactions(df)
        st.subheader("Filtros")
        search = st.text_input("Filtrar por descrição", placeholder="Digite parte da descrição")

        filtered = df.copy()
        if search:
            filtered = filtered[filtered["Description"].str.contains(search, case=False, na=False)]

        st.markdown("---")
        st.subheader("Tabela completa")
        st.dataframe(filtered, use_container_width=True)
        st.button("⬅️ Voltar", on_click=go_to_main)

