import io
import json
import os
import re
from typing import Iterable

import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    st.error("OPENAI_API_KEY was not loaded. Check .env file.")
    st.stop()

st.set_page_config(
    page_title="AI Bookkeeping Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

client = OpenAI(api_key=OPENAI_API_KEY)


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


def chunk_batches(items: Iterable, size: int) -> Iterable[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_ai_categorization(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Configure it in the .env file.")

    working_df = df.head(2000).copy()
    working_df["Date"] = working_df["Date"].astype(str)
    payload_rows = working_df.to_dict(orient="records")

    system_prompt = """
You are an expert US bookkeeper. Classify bank transactions for accounting systems (Zoho Books, QuickBooks). Infer business type from patterns in the data and choose appropriate expense/income accounts. For each transaction, return the fields: AI_Category, AI_Transaction_Type (Inflow, Outflow, Transfer), AI_Vendor, AI_Customer, AI_Account_Name, AI_Notes. Keep values concise and professional. Use Income/expense terminology, and align categories with typical US accounting.
"""

    ai_results: list[dict] = []
    for batch in chunk_batches(payload_rows, 30):
        user_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Classify the following transactions. Respond as JSON with a top-level key \"transactions\" "
                        "containing a list in the same order. Do not add extra commentary."
                    ),
                },
                {
                    "type": "text",
                    "text": json.dumps(batch, ensure_ascii=False),
                },
            ],
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, user_prompt],
            temperature=0.2,
        )

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content or "{}")
            ai_batch = parsed.get("transactions", [])
        except (json.JSONDecodeError, KeyError, AttributeError):
            ai_batch = []

        # Ensure alignment
        for original, enriched in zip(batch, ai_batch):
            merged = {**original, **enriched}
            ai_results.append(merged)

    if not ai_results:
        raise RuntimeError("AI classification returned no results.")

    enriched_df = pd.DataFrame(ai_results)
    for col in [
        "AI_Category",
        "AI_Transaction_Type",
        "AI_Vendor",
        "AI_Customer",
        "AI_Account_Name",
        "AI_Notes",
    ]:
        if col not in enriched_df.columns:
            enriched_df[col] = None

    return enriched_df


def generate_summary_text(df: pd.DataFrame) -> str:
    total_in = df.loc[df["Amount"] > 0, "Amount"].sum()
    total_out = df.loc[df["Amount"] < 0, "Amount"].sum()
    net = total_in + total_out

    top_expenses = (
        df[df["Amount"] < 0]
        .groupby("AI_Category")["Amount"]
        .sum()
        .sort_values()
        .head(5)
    )
    top_vendors = (
        df[df["Amount"] < 0]
        .groupby("AI_Vendor")["Amount"]
        .sum()
        .dropna()
        .sort_values()
        .head(5)
    )

    lines = [
        f"**Total inflows:** {format_currency(total_in)}",
        f"**Total outflows:** {format_currency(total_out)}",
        f"**Net result:** {format_currency(net)}",
        "",
        "**Top expense categories:**",
    ]
    if top_expenses.empty:
        lines.append("- No expenses identified.")
    else:
        for cat, val in top_expenses.items():
            cat_label = cat or "Uncategorized"
            lines.append(f"- {cat_label}: {format_currency(val)}")

    lines.append("\n**Top vendors by spend:**")
    if top_vendors.empty:
        lines.append("- No vendor expenses detected.")
    else:
        for vendor, val in top_vendors.items():
            vendor_label = vendor or "Unknown"
            lines.append(f"- {vendor_label}: {format_currency(val)}")

    suggestions = [
        "Review bank fees and recurring subscriptions for possible reductions.",
        "Consolidate vendor spending to negotiate better rates where feasible.",
        "Track large cash or transfer outflows to ensure proper documentation.",
    ]
    lines.append("\n**Suggestions:**")
    for tip in suggestions:
        lines.append(f"- {tip}")

    return "\n".join(lines)


def prepare_downloads(df: pd.DataFrame) -> tuple[str, str, str]:
    zoho_df = pd.DataFrame(
        {
            "Date": df["Date"],
            "Account": df["AI_Account_Name"],
            "Description": df["Description"],
            "Currency": "USD",
            "Amount": df["Amount"],
            "Contact Name": df["AI_Vendor"].fillna(df["AI_Customer"]),
            "Notes": df["AI_Notes"],
        }
    )

    qb_df = pd.DataFrame(
        {
            "Date": df["Date"],
            "Description": df["Description"],
            "Payee": df["AI_Vendor"].fillna(df["AI_Customer"]),
            "Memo": df["AI_Notes"],
            "Amount": df["Amount"],
            "Category": df["AI_Category"],
            "Account": df["AI_Account_Name"],
        }
    )

    vendors = (
        df["AI_Vendor"].dropna().drop_duplicates().reset_index(drop=True)
    )
    vendors_df = pd.DataFrame({"Vendor Name": vendors, "Notes": "Auto-generated from transactions"})

    return (
        zoho_df.to_csv(index=False),
        qb_df.to_csv(index=False),
        vendors_df.to_csv(index=False),
    )


if "page" not in st.session_state:
    st.session_state.page = "main"

if "ai_processed" not in st.session_state:
    st.session_state.ai_processed = False

if "ai_error" not in st.session_state:
    st.session_state.ai_error = None

page = st.session_state.page


# ------------------------------------------------------------
# Page: Summary (upload + metrics)
# ------------------------------------------------------------
if page == "main":
    render_header(
        "AI Bookkeeping Assistant",
        "Envie seu extrato, veja um resumo rápido e gere relatórios completos ao acionar a IA.",
    )

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
                st.session_state.df = df
                st.session_state.ai_processed = False
                st.session_state.ai_error = None

                summary = calculate_summary(df)
                render_metrics(summary)

                st.info("A análise com IA será executada apenas ao avançar para os detalhes.")

                if st.button("View AI Details & Bookkeeping Export", type="primary"):
                    st.session_state.page = "details"

        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Erro ao processar o arquivo: {exc}")
    else:
        st.info("Faça o upload do extrato para ver o resumo.")


# ------------------------------------------------------------
# Page: Details (AI analysis + exports)
# ------------------------------------------------------------
if page == "details":
    render_header(
        "📄 AI Details & Bookkeeping Export",
        "Classificação automática, relatórios e arquivos prontos para Zoho Books e QuickBooks.",
    )

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("Nenhum extrato carregado ainda. Volte para a página de Resumo e envie um arquivo.")
        if st.button("⬅️ Voltar"):
            st.session_state.page = "main"
    else:
        if not st.session_state.ai_processed:
            try:
                with st.spinner("Running AI analysis..."):
                    ai_df = run_ai_categorization(df)
                    st.session_state.df_ai = ai_df
                    st.session_state.ai_processed = True
                    st.session_state.ai_error = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.ai_error = str(exc)
                st.session_state.ai_processed = False

        if st.session_state.ai_error:
            st.error(f"Falha na classificação com IA: {st.session_state.ai_error}")
            if st.button("Tentar novamente"):
                st.session_state.ai_processed = False
                st.session_state.ai_error = None
                st.rerun()
        elif st.session_state.ai_processed:
            ai_df = st.session_state.df_ai
            st.success("IA concluída! Confira o relatório abaixo.")

            st.markdown("## AI Financial Overview")
            st.markdown(generate_summary_text(ai_df))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Despesas por categoria")
                expense_data = (
                    ai_df[ai_df["Amount"] < 0]
                    .groupby("AI_Category")["Amount"]
                    .sum()
                    .sort_values()
                )
                if not expense_data.empty:
                    st.bar_chart(expense_data)
                else:
                    st.info("Sem despesas para exibir.")

            with col2:
                st.markdown("### Entradas por categoria")
                income_data = (
                    ai_df[ai_df["Amount"] > 0]
                    .groupby("AI_Category")["Amount"]
                    .sum()
                    .sort_values(ascending=False)
                )
                if not income_data.empty:
                    st.bar_chart(income_data)
                else:
                    st.info("Sem entradas para exibir.")

            st.markdown("### Evolução do saldo")
            balance_df = ai_df.copy()
            balance_df["Date"] = pd.to_datetime(balance_df["Date"], errors="coerce")
            balance_df = balance_df.sort_values("Date")
            balance_df["Running Balance"] = balance_df["Amount"].cumsum()
            st.line_chart(balance_df.set_index("Date")["Running Balance"])

            st.markdown("---")
            st.markdown("### Tabela completa com IA")
            st.dataframe(
                ai_df[
                    [
                        "Date",
                        "Description",
                        "Amount",
                        "AI_Category",
                        "AI_Transaction_Type",
                        "AI_Vendor",
                        "AI_Customer",
                        "AI_Account_Name",
                        "AI_Notes",
                    ]
                ],
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("## Exportar")
            zoho_csv, qb_csv, vendors_csv = prepare_downloads(ai_df)

            colz, colq, colv = st.columns(3)
            colz.download_button(
                "Download Zoho Books file",
                data=zoho_csv,
                file_name="zoho_books_transactions.csv",
                mime="text/csv",
            )
            colq.download_button(
                "Download QuickBooks Transactions CSV",
                data=qb_csv,
                file_name="quickbooks_transactions.csv",
                mime="text/csv",
            )
            colv.download_button(
                "Download Vendors list (CSV)",
                data=vendors_csv,
                file_name="vendors.csv",
                mime="text/csv",
            )

            if len(df) > 2000:
                st.warning("Apenas as primeiras 2000 transações foram processadas pela IA.")

        if st.button("⬅️ Voltar"):
            st.session_state.page = "main"

