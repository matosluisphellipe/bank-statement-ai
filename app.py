import io
import json
import math
import re
from typing import Iterable

import altair as alt
import pandas as pd
import pdfplumber
import streamlit as st
from openai import OpenAI


try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(tr("api_key_missing"))
    st.stop()

st.set_page_config(
    page_title="AI Bookkeeping Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TRANSLATIONS = {
    "pt": {
        "app_title_main": "Assistente de Bookkeeping com IA",
        "app_subtitle_main": "Envie seu extrato, veja um resumo rápido e gere relatórios completos com IA.",
        "upload_section_title": "Upload do extrato",
        "upload_help": "Envie arquivos PDF, CSV, XLSX, TXT, OFX, QFX ou QBO com suas transações bancárias.",
        "summary_entries": "Total de entradas",
        "summary_exits": "Total de saídas",
        "summary_count": "Número de transações",
        "button_view_details": "Ver detalhes com IA & exportar",
        "details_title": "Detalhes com IA & Exportação de Bookkeeping",
        "details_subtitle": "Classificação automática, relatórios e arquivos prontos para Zoho Books e QuickBooks.",
        "no_file_info": "Nenhum extrato carregado ainda. Volte para a página principal e envie um arquivo.",
        "button_back": "⬅️ Voltar",
        "ai_running": "Rodando análise de IA...",
        "ai_done": "IA concluída! Confira o relatório abaixo.",
        "ai_failed": "Falha na classificação com IA",
        "ai_retry": "Tentar novamente",
        "ai_overview_title": "Visão financeira com IA",
        "chart_expenses_by_category": "Despesas por categoria",
        "chart_income_by_category": "Entradas por categoria",
        "chart_balance_evolution": "Evolução do saldo",
        "table_full_ai": "Tabela completa com IA",
        "export_section_title": "Exportar",
        "download_zoho": "Download arquivo Zoho Books",
        "download_qb": "Download QuickBooks Transactions CSV",
        "download_vendors": "Download Vendors list (CSV)",
        "ai_only_first_2000": "Apenas as primeiras 2000 transações foram processadas pela IA.",
        "file_processed_but_empty": "Arquivo processado, mas nenhuma transação foi identificada.",
        "upload_first_info": "Faça o upload do extrato para ver o resumo.",
        "theme_label": "Tema",
        "theme_light": "Claro",
        "theme_dark": "Escuro",
        "ai_progress_title": "Analisando transações com IA...",
        "ai_progress_estimate": "Estimativa de tempo: ~{seconds} segundos para {count} transações.",
        "ai_progress_batches": "Processando lote {current}/{total}...",
        "app_abbr": "Assistente IA",
        "upload_info": "A análise com IA será executada apenas ao avançar para os detalhes.",
        "upload_error": "❌ Erro ao processar o arquivo: {error}",
        "quota_error": "Limite de uso da API atingido. Verifique o faturamento do OpenAI ou utilize uma chave com créditos disponíveis.",
        "generic_error": "Falha ao chamar a API do OpenAI. Confira se a chave está correta e tente novamente.",
        "no_expenses": "Sem despesas para exibir.",
        "no_income": "Sem entradas para exibir.",
        "api_key_missing": "OPENAI_API_KEY não encontrado nas secrets do Streamlit.",
        "language_label": "Idioma",
        "category_label": "Categoria",
        "uncategorized": "Sem categoria",
    },
    "en": {
        "app_title_main": "AI Bookkeeping Assistant",
        "app_subtitle_main": "Upload your statement, see a quick summary and generate full AI-powered reports.",
        "upload_section_title": "Upload statement",
        "upload_help": "Upload PDF, CSV, XLSX, TXT, OFX, QFX or QBO bank statement files.",
        "summary_entries": "Total inflows",
        "summary_exits": "Total outflows",
        "summary_count": "Number of transactions",
        "button_view_details": "View AI details & bookkeeping export",
        "details_title": "AI Details & Bookkeeping Export",
        "details_subtitle": "Automatic classification, reports and ready-to-import files for Zoho Books and QuickBooks.",
        "no_file_info": "No statement loaded yet. Go back to the main page and upload a file.",
        "button_back": "⬅️ Back",
        "ai_running": "Running AI analysis...",
        "ai_done": "AI completed! Check the report below.",
        "ai_failed": "AI classification failed",
        "ai_retry": "Try again",
        "ai_overview_title": "AI Financial Overview",
        "chart_expenses_by_category": "Expenses by category",
        "chart_income_by_category": "Income by category",
        "chart_balance_evolution": "Balance evolution",
        "table_full_ai": "Full AI-enriched table",
        "export_section_title": "Export",
        "download_zoho": "Download Zoho Books file",
        "download_qb": "Download QuickBooks Transactions CSV",
        "download_vendors": "Download Vendors list (CSV)",
        "ai_only_first_2000": "Only the first 2000 transactions were processed by AI.",
        "file_processed_but_empty": "File processed, but no transactions were found.",
        "upload_first_info": "Upload a statement to see the summary.",
        "theme_label": "Theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "ai_progress_title": "Analyzing transactions with AI...",
        "ai_progress_estimate": "Estimated time: ~{seconds} seconds for {count} transactions.",
        "ai_progress_batches": "Processing batch {current}/{total}...",
        "app_abbr": "AI Assistant",
        "upload_info": "AI analysis will run only after moving to details.",
        "upload_error": "❌ Error processing the file: {error}",
        "quota_error": "API usage limit reached. Check OpenAI billing or use a key with available credits.",
        "generic_error": "Failed to call the OpenAI API. Confirm the key is correct and try again.",
        "no_expenses": "No expenses to display.",
        "no_income": "No income to display.",
        "api_key_missing": "OPENAI_API_KEY not found in Streamlit secrets.",
        "language_label": "Language",
        "category_label": "Category",
        "uncategorized": "Uncategorized",
    },
}


def tr(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "pt")
    text = TRANSLATIONS.get(lang, {}).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

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


def apply_theme_css(theme: str):
    palettes = {
        "light": {
            "bg": "#f7f8fa",
            "panel": "#ffffff",
            "border": "#e5e7eb",
            "text": "#0f172a",
            "muted": "#64748b",
            "accent": "linear-gradient(120deg, #1d4ed8 0%, #0ea5e9 100%)",
        },
        "dark": {
            "bg": "#0b1220",
            "panel": "#0f172a",
            "border": "#1f2937",
            "text": "#e2e8f0",
            "muted": "#94a3b8",
            "accent": "linear-gradient(120deg, #2563eb 0%, #22d3ee 100%)",
        },
    }

    colors = palettes.get(theme, palettes["light"])
    style = f"""
    <style>
    * {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {colors['bg']};
        color: {colors['text']};
    }}
    .block-container {{
        padding-top: 1rem;
        max-width: 1200px;
    }}
    [data-testid="stSidebar"], .stApp header {{
        background: {colors['panel']};
        color: {colors['text']};
    }}
    .top-banner {{
        background: {colors['panel']};
        border: 1px solid {colors['border']};
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        position: sticky;
        top: 0.5rem;
        z-index: 999;
    }}
    .section-card {{
        background: {colors['panel']};
        border: 1px solid {colors['border']};
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }}
    .stat-card {{
        background: {colors['panel']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    }}
    .stat-label {{
        color: {colors['muted']};
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }}
    .stat-value {{
        font-weight: 700;
        font-size: 1.25rem;
        color: {colors['text']};
    }}
    .pill-label {{
        color: {colors['muted']};
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
        display: block;
    }}
    .pill-control {{
        background: {colors['bg']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 0.4rem 0.75rem;
        width: 100%;
    }}
    .upload-helper {{
        color: {colors['muted']};
        font-size: 0.95rem;
    }}
    .stButton>button, .stDownloadButton>button {{
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: 1px solid {colors['border']};
        background: {colors['accent']};
        color: #fff;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    [data-testid="stMetricValue"] {{ color: {colors['text']}; }}
    [data-testid="stMetricLabel"] {{ color: {colors['muted']}; }}
    [data-testid="stToolbar"] {{ visibility: hidden; }}
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)


def prepare_category_totals(df: pd.DataFrame, positive: bool) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Amount"] = pd.to_numeric(cleaned["Amount"], errors="coerce")
    cleaned = cleaned.replace([math.inf, -math.inf], pd.NA).dropna(subset=["Amount"])
    cleaned["AI_Category"] = cleaned["AI_Category"].fillna(tr("uncategorized"))

    mask = cleaned["Amount"] > 0 if positive else cleaned["Amount"] < 0
    totals = (
        cleaned.loc[mask]
        .assign(Amount=lambda x: x["Amount"].abs())
        .groupby("AI_Category", dropna=False)["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"AI_Category": "Category", "Amount": "Total"})
        .sort_values("Total", ascending=False)
    )
    return totals


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
    with col1:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>{tr('summary_entries')}</div>
                <div class='stat-value'>{format_currency(summary['entries'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>{tr('summary_exits')}</div>
                <div class='stat-value'>{format_currency(summary['exits'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>{tr('summary_count')}</div>
                <div class='stat-value'>{summary['count']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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


def format_ai_error(error: Exception, lang: str) -> str:
    error_text = str(error)

    quota_keywords = ["insufficient_quota", "quota", "billing"]
    status_code = getattr(error, "status_code", None)
    if status_code == 429 or any(keyword in error_text for keyword in quota_keywords):
        return TRANSLATIONS.get(lang, TRANSLATIONS["pt"]).get("quota_error")

    return TRANSLATIONS.get(lang, TRANSLATIONS["pt"]).get("generic_error")


def run_ai_categorization(
    df: pd.DataFrame, progress_bar=None, status_placeholder=None
) -> pd.DataFrame:
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
    total_batches = math.ceil(len(payload_rows) / 30) if payload_rows else 0
    current_batch = 0
    for batch in chunk_batches(payload_rows, 30):
        current_batch += 1
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

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system_prompt}, user_prompt],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            lang = st.session_state.get("lang", "pt")
            raise RuntimeError(format_ai_error(exc, lang)) from exc

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

        if progress_bar is not None and total_batches:
            progress_bar.progress(current_batch / total_batches)
        if status_placeholder is not None and total_batches:
            status_placeholder.write(
                tr("ai_progress_batches", current=current_batch, total=total_batches)
            )

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


def generate_summary_text(df: pd.DataFrame, lang: str | None = None) -> str:
    lang = lang or st.session_state.get("lang", "pt")

    total_in = df.loc[df["Amount"] > 0, "Amount"].sum()
    total_out = df.loc[df["Amount"] < 0, "Amount"].sum()
    net = total_in + total_out

    income_categories = (
        df[df["Amount"] > 0]
        .groupby("AI_Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
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

    if lang == "en":
        header_lines = [
            f"**Total inflows:** {format_currency(total_in)}",
            f"**Total outflows:** {format_currency(total_out)}",
            f"**Net result:** {format_currency(net)}",
            "",
            "**Revenue highlights:**",
        ]
        if income_categories.empty:
            header_lines.append("- No income categories identified.")
        else:
            for cat, val in income_categories.items():
                cat_label = cat or "Uncategorized"
                header_lines.append(f"- {cat_label}: {format_currency(val)}")

        header_lines.extend([
            "\n**Top expense categories:**",
        ])
        if top_expenses.empty:
            header_lines.append("- No expenses identified.")
        else:
            for cat, val in top_expenses.items():
                cat_label = cat or "Uncategorized"
                header_lines.append(f"- {cat_label}: {format_currency(val)}")

        header_lines.append("\n**Top vendors by spend:**")
        if top_vendors.empty:
            header_lines.append("- No vendor expenses detected.")
        else:
            for vendor, val in top_vendors.items():
                vendor_label = vendor or "Unknown"
                header_lines.append(f"- {vendor_label}: {format_currency(val)}")

        suggestions = [
            "Diversify revenue sources to avoid dependence on a few customers.",
            "Strengthen recurring revenue streams through renewals or long-term contracts.",
            "Review bank fees and recurring subscriptions for possible reductions.",
            "Consolidate vendor spending to negotiate better rates where feasible.",
            "Track large cash or transfer outflows to ensure proper documentation.",
        ]
        result_comment = (
            f"The business reported a net result of {format_currency(net)}, "
            "highlighting overall cash position for the analyzed period."
        )
        suggestions_title = "**Suggestions:**"
    else:
        header_lines = [
            f"**Total de entradas:** {format_currency(total_in)}",
            f"**Total de saídas:** {format_currency(total_out)}",
            f"**Resultado líquido:** {format_currency(net)}",
            "",
            "**Destaques de receita:**",
        ]
        if income_categories.empty:
            header_lines.append("- Nenhuma categoria de receita identificada.")
        else:
            for cat, val in income_categories.items():
                cat_label = cat or "Sem categoria"
                header_lines.append(f"- {cat_label}: {format_currency(val)}")

        header_lines.extend([
            "\n**Principais categorias de despesa:**",
        ])
        if top_expenses.empty:
            header_lines.append("- Nenhuma despesa identificada.")
        else:
            for cat, val in top_expenses.items():
                cat_label = cat or "Sem categoria"
                header_lines.append(f"- {cat_label}: {format_currency(val)}")

        header_lines.append("\n**Top fornecedores por gasto:**")
        if top_vendors.empty:
            header_lines.append("- Nenhuma despesa por fornecedor detectada.")
        else:
            for vendor, val in top_vendors.items():
                vendor_label = vendor or "Desconhecido"
                header_lines.append(f"- {vendor_label}: {format_currency(val)}")

        suggestions = [
            "Avalie a concentração de receitas em poucos clientes e diversifique quando possível.",
            "Reforce contratos recorrentes para manter a previsibilidade de entradas.",
            "Revise tarifas bancárias e assinaturas recorrentes para possíveis reduções.",
            "Consolide gastos com fornecedores para negociar melhores condições.",
            "Monitore saídas em dinheiro ou transferências para garantir documentação adequada.",
        ]
        result_comment = (
            f"A empresa apresentou um resultado líquido de {format_currency(net)}, "
            "refletindo a posição de caixa do período analisado."
        )
        suggestions_title = "**Sugestões:**"

    lines = header_lines + ["", f"**{result_comment}**", "", suggestions_title]
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


if "lang" not in st.session_state:
    st.session_state.lang = "pt"

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "page" not in st.session_state:
    st.session_state.page = "main"

if "ai_processed" not in st.session_state:
    st.session_state.ai_processed = False

if "ai_error" not in st.session_state:
    st.session_state.ai_error = None

current_lang = st.session_state.get("lang", "pt")
current_theme = st.session_state.get("theme", "light")

with st.container():
    st.markdown("<div class='top-banner'>", unsafe_allow_html=True)
    col_app, col_lang, col_theme = st.columns([1.5, 1, 1])
    with col_app:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:#4b5563;'>{tr('app_abbr')}</div>",
            unsafe_allow_html=True,
        )
    with col_lang:
        lang_selection = st.selectbox(
            tr("language_label"),
            options=["pt", "en"],
            format_func=lambda x: "Português" if x == "pt" else "English",
            index=0 if current_lang == "pt" else 1,
            key="lang_selector",
        )
        if lang_selection != current_lang:
            st.session_state.lang = lang_selection
            st.rerun()

    with col_theme:
        theme_choice = st.selectbox(
            tr("theme_label"),
            [tr("theme_light"), tr("theme_dark")],
            index=0 if current_theme == "light" else 1,
            key="theme_selector",
        )
        st.session_state.theme = "light" if theme_choice == tr("theme_light") else "dark"
    st.markdown("</div>", unsafe_allow_html=True)

apply_theme_css(st.session_state.theme)

page = st.session_state.page


# ------------------------------------------------------------
# Page: Summary (upload + metrics)
# ------------------------------------------------------------
if page == "main":
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        render_header(tr("app_title_main"), tr("app_subtitle_main"))
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader(tr("upload_section_title"))
        st.markdown(f"<p class='upload-helper'>{tr('upload_help')}</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            tr("upload_help"),
            type=["pdf", "csv", "xlsx", "txt", "ofx", "qfx", "qbo"],
            help=tr("upload_help"),
        )

        if uploaded:
            try:
                df = normalize_transactions(parse_file(uploaded))
                if df.empty:
                    st.warning(tr("file_processed_but_empty"))
                else:
                    df = df.dropna(axis=1, how="all")
                    st.session_state.df = df
                    st.session_state.ai_processed = False
                    st.session_state.ai_error = None

                    summary = calculate_summary(df)
                    with st.container():
                        render_metrics(summary)

                    st.info(tr("upload_info"))

                    st.write("")
                    if st.button(tr("button_view_details"), type="primary"):
                        st.session_state.page = "details"

            except Exception as exc:  # noqa: BLE001
                st.error(tr("upload_error", error=exc))
        else:
            st.info(tr("upload_first_info"))
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Page: Details (AI analysis + exports)
# ------------------------------------------------------------
if page == "details":
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        render_header(tr("details_title"), tr("details_subtitle"))
        st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info(tr("no_file_info"))
        if st.button(tr("button_back")):
            st.session_state.page = "main"
    else:
        if not st.session_state.ai_processed:
            count = min(len(df), 2000)
            batch_size = 30
            total_batches = math.ceil(count / batch_size)
            estimated_seconds = total_batches * 2

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### " + tr("ai_progress_title"))
            st.markdown(tr("ai_progress_estimate", seconds=estimated_seconds, count=count))
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            try:
                ai_df = run_ai_categorization(
                    df, progress_bar=progress_bar, status_placeholder=status_placeholder
                )
                progress_bar.progress(1.0)
                status_placeholder.write(tr("ai_done"))
                st.session_state.df_ai = ai_df
                st.session_state.ai_processed = True
                st.session_state.ai_error = None
            except Exception as exc:  # noqa: BLE001
                if "progress_bar" in locals():
                    progress_bar.empty()
                if "status_placeholder" in locals():
                    status_placeholder.empty()
                st.session_state.ai_error = str(exc)
                st.session_state.ai_processed = False
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.ai_error:
            st.error(f"{tr('ai_failed')}: {st.session_state.ai_error}")
            if st.button(tr("ai_retry")):
                st.session_state.ai_processed = False
                st.session_state.ai_error = None
                st.rerun()
        elif st.session_state.ai_processed:
            ai_df = st.session_state.df_ai
            st.success(tr("ai_done"))

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("## " + tr("ai_overview_title"))
            st.markdown(generate_summary_text(ai_df, lang=st.session_state.get("lang", "pt")))
            st.markdown("</div>", unsafe_allow_html=True)

            charts_container = st.container()
            with charts_container:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### " + tr("chart_expenses_by_category"))
                    expense_totals = prepare_category_totals(ai_df, positive=False)
                    if not expense_totals.empty:
                        expense_chart = (
                            alt.Chart(expense_totals)
                            .mark_bar(cornerRadiusEnd=6)
                            .encode(
                                x=alt.X("Total:Q", title=tr("summary_exits")),
                                y=alt.Y("Category:N", sort="-x", title=tr("category_label")),
                                color=alt.Color(
                                    "Total:Q", scale=alt.Scale(scheme="reds"), legend=None
                                ),
                                tooltip=["Category", "Total"],
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(expense_chart, use_container_width=True)
                    else:
                        st.info(tr("no_expenses"))

                with col2:
                    st.markdown("### " + tr("chart_income_by_category"))
                    income_totals = prepare_category_totals(ai_df, positive=True)
                    if not income_totals.empty:
                        income_chart = (
                            alt.Chart(income_totals)
                            .mark_bar(cornerRadiusEnd=6)
                            .encode(
                                x=alt.X("Category:N", sort="-y", title=tr("category_label")),
                                y=alt.Y("Total:Q", title=tr("summary_entries")),
                                color=alt.Color("Total:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                tooltip=["Category", "Total"],
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(income_chart, use_container_width=True)
                    else:
                        st.info(tr("no_income"))
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### " + tr("chart_balance_evolution"))
            balance_df = ai_df.copy()
            balance_df["Amount"] = pd.to_numeric(balance_df["Amount"], errors="coerce")
            balance_df["Date"] = pd.to_datetime(balance_df["Date"], errors="coerce")
            balance_df = balance_df.replace([math.inf, -math.inf], pd.NA).dropna(subset=["Date", "Amount"])
            balance_df = balance_df.sort_values("Date")
            balance_df["Running Balance"] = balance_df["Amount"].cumsum()
            if not balance_df.empty:
                balance_chart = (
                    alt.Chart(balance_df)
                    .mark_line(point=True, interpolate="monotone")
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y(
                            "Running Balance:Q",
                            title="Running Balance",
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=["Date:T", "Running Balance:Q"],
                    )
                    .interactive()
                    .properties(height=400)
                )
                st.altair_chart(balance_chart, use_container_width=True)
            else:
                st.info(tr("file_processed_but_empty"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### " + tr("table_full_ai"))
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
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### " + tr("export_section_title"))
            st.markdown("Prepare arquivos para Zoho Books, QuickBooks e vendors list.")
            zoho_csv, qb_csv, vendors_csv = prepare_downloads(ai_df)

            colz, colq, colv = st.columns(3)
            colz.download_button(
                tr("download_zoho"),
                data=zoho_csv,
                file_name="zoho_books_transactions.csv",
                mime="text/csv",
            )
            colq.download_button(
                tr("download_qb"),
                data=qb_csv,
                file_name="quickbooks_transactions.csv",
                mime="text/csv",
            )
            colv.download_button(
                tr("download_vendors"),
                data=vendors_csv,
                file_name="vendors.csv",
                mime="text/csv",
            )

            if len(df) > 2000:
                st.warning(tr("ai_only_first_2000"))

        if st.button(tr("button_back")):
            st.session_state.page = "main"

