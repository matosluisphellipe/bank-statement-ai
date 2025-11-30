import io
import json
import math
import re
from typing import Iterable

import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Bookkeeping Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LANG = {
    "en": {
        "app_name": "AI Bookkeeping Assistant",
        "app_tagline": "Modern AI-powered bookkeeping with instant insights and exports.",
        "upload_title": "Upload statement",
        "upload_help": "Upload PDF, CSV, XLSX, TXT, OFX, QFX or QBO bank statement files.",
        "upload_cta": "Analyze with AI",
        "upload_info": "AI analysis only runs when you continue to details.",
        "upload_first_info": "Upload a statement to see the summary.",
        "summary_entries": "Total inflows",
        "summary_exits": "Total outflows",
        "summary_count": "Transactions",
        "details_title": "AI details & bookkeeping export",
        "details_subtitle": "Automatic classification, elegant visuals and ready-to-import files.",
        "no_file_info": "No statement loaded yet. Go back and upload a file.",
        "ai_progress_title": "Analyzing transactions with AI...",
        "ai_progress_start": "Starting analysis...",
        "ai_progress_sending": "Sending transactions to AI...",
        "ai_progress_batch": "Processing batch {current} of {total}...",
        "ai_progress_almost": "Almost there...",
        "ai_done": "AI completed! Check the report below.",
        "ai_failed": "AI classification failed",
        "ai_retry": "Try again",
        "ai_overview": "Financial overview",
        "chart_expenses": "💸 Expenses by category",
        "chart_income": "💰 Income by category",
        "chart_balance": "📈 Balance evolution",
        "table_full": "AI-enriched table",
        "export_section": "Exports",
        "download_zoho": "⬇️ Zoho Books",
        "download_qb": "📥 QuickBooks CSV",
        "download_vendors": "🏷️ Vendor list",
        "ai_only_first_2000": "Only the first 2000 transactions were processed by AI.",
        "file_empty": "File processed, but no transactions were found.",
        "language_label": "Language",
        "theme_label": "Theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "category_label": "Category",
        "uncategorized": "Uncategorized",
        "report_business_type": "Business type detected",
        "report_cashflow": "Cashflow summary",
        "report_income": "Top income sources",
        "report_expenses": "Top expense categories",
        "report_observations": "Observations",
        "report_suggestions": "AI suggestions",
        "hero_primary": "Financial clarity in seconds",
        "hero_secondary": "Upload, classify and export with a polished corporate SaaS experience.",
        "button_back": "⬅️ Back",
        "no_expenses": "No expenses to display.",
        "no_income": "No income to display.",
        "upload_error": "❌ Error processing the file: {error}",
        "quota_error": "API usage limit reached. Check OpenAI billing or use a key with available credits.",
        "generic_error": "Failed to call the OpenAI API. Confirm the key is correct and try again.",
        "api_key_missing": "OPENAI_API_KEY not found in Streamlit secrets.",
    },
    "pt": {
        "app_name": "Assistente de Bookkeeping com IA",
        "app_tagline": "Bookkeeping moderno com IA, insights instantâneos e exports prontos.",
        "upload_title": "Upload do extrato",
        "upload_help": "Envie arquivos PDF, CSV, XLSX, TXT, OFX, QFX ou QBO com suas transações bancárias.",
        "upload_cta": "Analisar com IA",
        "upload_info": "A análise com IA é executada apenas ao avançar para os detalhes.",
        "upload_first_info": "Faça o upload do extrato para ver o resumo.",
        "summary_entries": "Total de entradas",
        "summary_exits": "Total de saídas",
        "summary_count": "Transações",
        "details_title": "Detalhes com IA & exportação",
        "details_subtitle": "Classificação automática, visuais elegantes e arquivos prontos para importar.",
        "no_file_info": "Nenhum extrato carregado ainda. Volte e envie um arquivo.",
        "ai_progress_title": "Analisando transações com IA...",
        "ai_progress_start": "Iniciando análise...",
        "ai_progress_sending": "Enviando transações para IA...",
        "ai_progress_batch": "Aguarde, processando lote {current} de {total}...",
        "ai_progress_almost": "Quase lá...",
        "ai_done": "IA concluída! Confira o relatório abaixo.",
        "ai_failed": "Falha na classificação com IA",
        "ai_retry": "Tentar novamente",
        "ai_overview": "Visão financeira",
        "chart_expenses": "💸 Despesas por categoria",
        "chart_income": "💰 Entradas por categoria",
        "chart_balance": "📈 Evolução do saldo",
        "table_full": "Tabela completa com IA",
        "export_section": "Exportar",
        "download_zoho": "⬇️ Zoho Books",
        "download_qb": "📥 QuickBooks CSV",
        "download_vendors": "🏷️ Lista de fornecedores",
        "ai_only_first_2000": "Apenas as primeiras 2000 transações foram processadas pela IA.",
        "file_empty": "Arquivo processado, mas nenhuma transação foi identificada.",
        "language_label": "Idioma",
        "theme_label": "Tema",
        "theme_light": "Claro",
        "theme_dark": "Escuro",
        "category_label": "Categoria",
        "uncategorized": "Sem categoria",
        "report_business_type": "Tipo de negócio detectado",
        "report_cashflow": "Resumo de caixa",
        "report_income": "Principais fontes de receita",
        "report_expenses": "Principais categorias de despesa",
        "report_observations": "Observações",
        "report_suggestions": "Sugestões da IA",
        "hero_primary": "Clareza financeira em segundos",
        "hero_secondary": "Envie, classifique e exporte com uma experiência corporativa moderna.",
        "button_back": "⬅️ Voltar",
        "no_expenses": "Sem despesas para exibir.",
        "no_income": "Sem entradas para exibir.",
        "upload_error": "❌ Erro ao processar o arquivo: {error}",
        "quota_error": "Limite de uso da API atingido. Verifique o faturamento do OpenAI ou utilize uma chave com créditos disponíveis.",
        "generic_error": "Falha ao chamar a API do OpenAI. Confira se a chave está correta e tente novamente.",
        "api_key_missing": "OPENAI_API_KEY não encontrado nas secrets do Streamlit.",
    },
}


if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "page" not in st.session_state:
    st.session_state.page = "main"
if "ai_processed" not in st.session_state:
    st.session_state.ai_processed = False
if "ai_error" not in st.session_state:
    st.session_state.ai_error = None

t = LANG[st.session_state.lang]


def tr(key: str, **kwargs) -> str:
    text = t.get(key, key)
    return text.format(**kwargs) if kwargs else text


def apply_theme_css():
    theme = st.session_state.get("theme", "light")
    palettes = {
        "light": {
            "bg": "#f8fafc",
            "panel": "#ffffff",
            "text": "#111827",
            "muted": "#6b7280",
            "border": "#e5e7eb",
            "accent": "#0066ff",
        },
        "dark": {
            "bg": "#0d1117",
            "panel": "#161b22",
            "text": "#e6edf3",
            "muted": "#a1acc0",
            "border": "#1f2937",
            "accent": "#238dff",
        },
    }
    colors = palettes[theme]
    gradient = "linear-gradient(120deg, #0e7490 0%, #0a2540 50%, #0f4c75 100%)"

    style = f"""
    <style>
    :root {{
        --bg: {colors['bg']};
        --panel: {colors['panel']};
        --text: {colors['text']};
        --muted: {colors['muted']};
        --border: {colors['border']};
        --accent: {colors['accent']};
    }}
    * {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
    [data-testid="stAppViewContainer"] {{
        background: var(--bg);
        color: var(--text);
    }}
    .block-container {{
        max-width: 1180px;
        padding-top: 1.25rem;
    }}
    .navbar {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
        position: sticky;
        top: 0.5rem;
        z-index: 100;
    }}
    .hero-card, .section-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 14px 40px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.75rem;
    }}
    .stat-card {{
        background: linear-gradient(180deg, rgba(0,0,0,0.03), rgba(0,0,0,0)) var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 10px 32px rgba(0,0,0,0.07);
    }}
    .stat-label {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 0.2rem; }}
    .stat-value {{ color: var(--text); font-weight: 700; font-size: 1.3rem; }}
    .pill {{
        background: rgba(0,0,0,0.04);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        width: 100%;
    }}
    .download-row {{ display: flex; gap: 0.6rem; justify-content: center; }}
    .download-row .stDownloadButton>button, .stDownloadButton>button {{
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        border: 1px solid var(--border);
        background: {gradient};
        color: #fff;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }}
    .download-row .stDownloadButton>button:hover, .stDownloadButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(0,0,0,0.2);
    }}
    .progress-shell {{
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem;
        background: var(--panel);
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }}
    .progress-bar {{
        position: relative;
        height: 10px;
        background: rgba(0,0,0,0.08);
        border-radius: 999px;
        overflow: hidden;
    }}
    .progress-bar span {{
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 0;
        border-radius: 999px;
        background: linear-gradient(90deg, #0ea5e9 0%, #0066ff 60%, #0f4c75 100%);
        box-shadow: 0 10px 30px rgba(0,102,255,0.35);
        transition: width 300ms ease;
    }}
    .progress-text {{
        margin-top: 0.5rem;
        color: var(--muted);
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .report-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.8rem;
        margin-top: 0.5rem;
    }}
    .report-card {{
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(35,141,255,0.06), rgba(0,0,0,0)) var(--panel);
        box-shadow: 0 12px 28px rgba(0,0,0,0.08);
    }}
    .divider {{
        border-bottom: 1px solid var(--border);
        margin: 0.75rem 0 0.5rem;
    }}
    .styled-table tbody tr:nth-child(even) {{
        background: rgba(0,0,0,0.03);
    }}
    .styled-table tbody tr:hover {{
        background: rgba(0,102,255,0.08);
    }}
    .styled-table table {{ border-collapse: separate; border-spacing: 0; }}
    .styled-table td, .styled-table th {{ border: 1px solid var(--border); padding: 8px; }}
    .styled-table th {{ background: rgba(0,102,255,0.06); color: var(--text); font-weight: 700; }}
    [data-testid="stToolbar"] {{ visibility: hidden; }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(tr("api_key_missing"))
    st.stop()

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
    st.markdown("<div class='stat-grid'>", unsafe_allow_html=True)
    for label, value in [
        (tr("summary_entries"), format_currency(summary["entries"])),
        (tr("summary_exits"), format_currency(summary["exits"])),
        (tr("summary_count"), f"{summary['count']:,}"),
    ]:
        st.markdown(
            f"""
            <div class='stat-card'>
                <div class='stat-label'>{label}</div>
                <div class='stat-value'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


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
        return LANG.get(lang, LANG["en"]).get("quota_error")

    return LANG.get(lang, LANG["en"]).get("generic_error")


def infer_business_type(df: pd.DataFrame) -> str:
    text_blob = " ".join(
        df.get("Description", pd.Series(dtype=str)).fillna("").astype(str).str.lower().tolist()
    )
    vendor_blob = " ".join(df.get("AI_Vendor", pd.Series(dtype=str)).fillna("").astype(str).str.lower().tolist())
    signals = text_blob + " " + vendor_blob
    mapping = {
        "home depot": "Construction",
        "lowe": "Construction",
        "rk miles": "Construction",
        "dumpster": "Construction",
        "waste": "Construction",
        "hvac": "HVAC",
        "plumbing": "Construction",
        "auto": "Auto Repair",
        "fleet": "Trucking",
        "trucking": "Trucking",
        "fuel": "Trucking",
        "lawn": "Landscaping",
        "landscap": "Landscaping",
        "clean": "Cleaning",
        "janitorial": "Cleaning",
        "marketing": "Services",
        "consult": "Professional Services",
    }
    for keyword, industry in mapping.items():
        if keyword in signals:
            return industry
    return "General Business"


def validate_ai_results(enriched_df: pd.DataFrame) -> pd.DataFrame:
    df = enriched_df.copy()
    business_type = infer_business_type(df)

    vendor_rules = [
        {"keywords": ["autozone"], "category": "Auto Parts & Vehicle Maintenance", "account": "Vehicle Repairs"},
        {"keywords": ["gulf", "mobil", "circle k", "shell", "chevron", "bp"], "category": "Fuel", "account": "Fuel"},
        {
            "keywords": ["home depot", "lowe", "rk miles", "menards"],
            "category": "Building Materials / Construction Supplies",
            "account": "Materials",
        },
        {"keywords": ["star waste", "waste", "dumpster"], "category": "Waste Removal / Dumpster Fees", "account": "Waste Removal"},
        {"keywords": ["fleetio"], "category": "Fleet Management Software", "account": "Software Subscriptions"},
        {"keywords": ["connecteam"], "category": "Employee Management Software", "account": "Software Subscriptions"},
        {"keywords": ["thumbtack"], "category": "Advertising & Marketing", "account": "Advertising"},
    ]

    for idx, row in df.iterrows():
        desc = str(row.get("Description") or "").lower()
        vendor = str(row.get("AI_Vendor") or "").lower()
        notes = str(row.get("AI_Notes") or "").strip()
        amount = row.get("Amount")

        if pd.notna(amount):
            if amount > 0 and row.get("AI_Transaction_Type") not in {"Inflow", "Transfer"}:
                df.at[idx, "AI_Transaction_Type"] = "Inflow"
                notes += " | Validation: adjusted to inflow based on positive amount."
            elif amount < 0 and row.get("AI_Transaction_Type") not in {"Outflow", "Transfer"}:
                df.at[idx, "AI_Transaction_Type"] = "Outflow"
                notes += " | Validation: adjusted to outflow based on negative amount."

        for rule in vendor_rules:
            if any(keyword in vendor or keyword in desc for keyword in rule["keywords"]):
                if row.get("AI_Category") != rule["category"]:
                    df.at[idx, "AI_Category"] = rule["category"]
                    df.at[idx, "AI_Account_Name"] = rule["account"]
                    notes += f" | Validation: vendor matched {rule['category']} guidance."

        if "amazon" in vendor and not row.get("AI_Category"):
            df.at[idx, "AI_Category"] = "Office Supplies"
            df.at[idx, "AI_Account_Name"] = "Office Supplies"
            notes += " | Validation: Amazon purchase defaulted to office supplies; refine if details available."

        large_txn = pd.notna(amount) and abs(amount) >= 5000
        if large_txn and not row.get("AI_Category"):
            if amount > 0:
                df.at[idx, "AI_Category"] = "Owner Contribution"
                df.at[idx, "AI_Account_Name"] = "Owner Equity"
                notes += " | Validation: large inflow tagged as potential owner contribution."
            else:
                fallback = "Materials" if business_type in {"Construction", "HVAC", "Landscaping"} else "Contractors"
                df.at[idx, "AI_Category"] = fallback
                df.at[idx, "AI_Account_Name"] = fallback
                notes += " | Validation: large outflow aligned with core operations."

        if business_type == "Construction" and row.get("AI_Category") == "Fuel" and "autozone" in vendor:
            df.at[idx, "AI_Category"] = "Auto Parts & Vehicle Maintenance"
            df.at[idx, "AI_Account_Name"] = "Vehicle Repairs"
            notes += " | Validation: AutoZone should be vehicle maintenance, not fuel."

        df.at[idx, "AI_Notes"] = notes.strip(" |")

    df["AI_Business_Type"] = business_type
    return df


def run_ai_categorization(
    df: pd.DataFrame, progress_placeholder=None, status_placeholder=None
) -> pd.DataFrame:
    if df.empty:
        return df

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Configure it in the .env file.")

    working_df = df.head(2000).copy()
    working_df["Date"] = working_df["Date"].astype(str)
    payload_rows = working_df.to_dict(orient="records")

    system_prompt = """
You are a senior U.S. accountant and forensic bookkeeper with expertise in classifying financial transactions for businesses across all industries. Your job is to classify each transaction with extremely high accuracy.

You MUST:
1. Read each transaction holistically: description, vendor name, amount, patterns, frequency.
2. Infer the type of business (construction, cleaning, trucking, auto repair, HVAC, landscaping, retail, services, professional consulting, etc.) by analyzing:
   - Vendors
   - Spending categories
   - Tools or materials purchased
   - Fuel patterns
   - Payroll patterns
   - Home Depot / Lowe’s usage
   - Commercial waste services
   - Thumbtack ads
   - Vehicle maintenance or rental
   - Office expenses

3. Assign categories CONSISTENT with U.S. GAAP, IRS guidelines, and standard chart of accounts used by:
   - QuickBooks Online
   - Zoho Books

4. Vendor intelligence:
   - AutoZone = Auto Parts & Vehicle Maintenance (NEVER fuel)
   - Gulf, Mobil, Circle K, Shell = Fuel / Gasoline
   - Home Depot, Lowe’s, RK Miles = Building Materials / Construction Supplies
   - Star Waste Systems = Waste Removal / Dumpster Fees
   - Fleetio = Fleet Management Software
   - Connecteam = Employee management / workforce software
   - Thumbtack = Advertising & Marketing
   - Amazon = Inspect context to determine category: tools, electronics, materials, office supplies, etc.

5. For each transaction, return:
   - AI_Category
   - AI_Transaction_Type (Inflow, Outflow, Transfer)
   - AI_Vendor (normalized vendor name)
   - AI_Customer (if applicable)
   - AI_Account_Name (specific accounting account)
   - AI_Notes (short explanation of why it was classified this way)

6. NEVER guess randomly. If information is incomplete, infer using:
   - Vendor database context
   - U.S. business logic
   - Industry patterns
   - Description keywords
   - Historical patterns in the file

7. Match inflows with probable revenue sources:
   - Zelle from individuals = revenue, loan repayment, or owner contribution depending on pattern
   - ACH deposits = revenue unless clearly payroll or transfer

8. Output must be JSON only with no extra commentary.
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
            lang = st.session_state.get("lang", "en")
            raise RuntimeError(format_ai_error(exc, lang)) from exc

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content or "{}")
            ai_batch = parsed.get("transactions", [])
        except (json.JSONDecodeError, KeyError, AttributeError):
            ai_batch = []

        for original, enriched in zip(batch, ai_batch):
            merged = {**original, **enriched}
            ai_results.append(merged)

        if progress_placeholder is not None and total_batches:
            percent = current_batch / total_batches
            progress_placeholder.markdown(
                f"<div class='progress-bar'><span style='width:{percent*100:.1f}%'></span></div>",
                unsafe_allow_html=True,
            )
        if status_placeholder is not None and total_batches:
            if current_batch >= total_batches:
                message = tr("ai_progress_almost")
            else:
                message = tr("ai_progress_batch", current=current_batch, total=total_batches)
            status_placeholder.markdown(f"<div class='progress-text'>{message}</div>", unsafe_allow_html=True)

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

    validated_df = validate_ai_results(enriched_df)
    return validated_df


def generate_report_cards(df: pd.DataFrame):
    income_df = df[df["Amount"] > 0].copy()
    expense_df = df[df["Amount"] < 0].copy()

    business_type = df.get("AI_Business_Type")
    business_label = business_type.iloc[0] if isinstance(business_type, pd.Series) and not business_type.empty else "General"

    income_categories = (
        income_df.groupby("AI_Category")["Amount"].sum().sort_values(ascending=False).head(5)
    )
    expense_categories = (
        expense_df.groupby("AI_Category")["Amount"].sum().sort_values().head(5)
    )
    top_income_sources = (
        income_df.groupby("AI_Customer")["Amount"].sum().dropna().sort_values(ascending=False).head(5)
    )
    top_vendors = (
        expense_df.groupby("AI_Vendor")["Amount"].sum().dropna().sort_values().head(5)
    )

    total_in = income_df["Amount"].sum()
    total_out = expense_df["Amount"].sum()
    net = total_in + total_out

    cards = [
        {
            "title": f"🏢 {tr('report_business_type')}",
            "content": f"<strong>{business_label}</strong>",
        },
        {
            "title": f"💹 {tr('report_cashflow')}",
            "content": f"<div>In: {format_currency(total_in)}</div><div>Out: {format_currency(total_out)}</div><div><strong>Net: {format_currency(net)}</strong></div>",
        },
        {
            "title": f"💼 {tr('report_income')}",
            "content": "<br>".join(
                [f"{cat or tr('uncategorized')}: {format_currency(val)}" for cat, val in income_categories.items()]
            )
            or tr("no_income"),
        },
        {
            "title": f"🧾 {tr('report_expenses')}",
            "content": "<br>".join(
                [f"{cat or tr('uncategorized')}: {format_currency(val)}" for cat, val in expense_categories.items()]
            )
            or tr("no_expenses"),
        },
        {
            "title": f"🔗 {tr('report_observations')}",
            "content": "<ul>"
            + "".join(
                [
                    f"<li>{vendor or 'Unknown'}: {format_currency(val)}</li>"
                    for vendor, val in top_vendors.items()
                ]
            )
            + "</ul>" if not top_vendors.empty else tr("no_expenses"),
        },
    ]

    suggestions = [
        "Align revenue recognition with recurring payer patterns to improve forecasting.",
        "Review high spend vendors for contract renegotiation and alignment with business type.",
        "Tag owner contributions and loan repayments distinctly to keep equity and liabilities clean.",
    ]
    if business_label in {"Construction", "HVAC", "Landscaping"}:
        suggestions.append("Track materials vs. subcontractors separately to protect gross margin.")
    elif business_label in {"Cleaning", "Professional Services"}:
        suggestions.append("Separate labor, supplies, and advertising to monitor client acquisition costs.")
    else:
        suggestions.append("Maintain clear segregation between operating expenses and discretionary spending.")

    cards.append(
        {
            "title": f"✨ {tr('report_suggestions')}",
            "content": "<ul>" + "".join([f"<li>{tip}</li>" for tip in suggestions]) + "</ul>",
        }
    )

    st.markdown("<div class='report-grid'>", unsafe_allow_html=True)
    for card in cards:
        st.markdown(
            f"""
            <div class='report-card'>
                <div style='font-weight:700;margin-bottom:0.35rem;'>{card['title']}</div>
                <div style='color:var(--text);font-size:0.98rem;'>{card['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


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


# ------------------------------------------------------------
# Layout components
# ------------------------------------------------------------
apply_theme_css()

with st.container():
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            f"<div style='font-size:18px;font-weight:800;letter-spacing:0.01em;'>{tr('app_name')}</div>"
            f"<div style='color:var(--muted);font-size:13px;'>{tr('app_tagline')}</div>",
            unsafe_allow_html=True,
        )
    with right:
        lang = st.selectbox(tr("language_label"), options=["en", "pt"], index=0 if st.session_state.lang == "en" else 1)
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            st.rerun()
        theme_choice = st.selectbox(
            tr("theme_label"),
            options=[tr("theme_light"), tr("theme_dark")],
            index=0 if st.session_state.theme == "light" else 1,
            key="theme_selector",
        )
        st.session_state.theme = "light" if theme_choice == tr("theme_light") else "dark"
    st.markdown("</div>", unsafe_allow_html=True)

apply_theme_css()

page = st.session_state.page

# ------------------------------------------------------------
# Page: Summary (upload + metrics)
# ------------------------------------------------------------
if page == "main":
    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:30px;font-weight:800;color:var(--text);'>{tr('hero_primary')}</div>"
        f"<div style='color:var(--muted);margin-top:6px;font-size:16px;'>{tr('hero_secondary')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"### {tr('upload_title')}")

    uploaded = st.file_uploader(
        tr("upload_help"),
        type=["pdf", "csv", "xlsx", "txt", "ofx", "qfx", "qbo"],
        help=tr("upload_help"),
    )

    if uploaded:
        try:
            df = normalize_transactions(parse_file(uploaded))
            if df.empty:
                st.warning(tr("file_empty"))
            else:
                df = df.dropna(axis=1, how="all")
                st.session_state.df = df
                st.session_state.ai_processed = False
                st.session_state.ai_error = None

                summary = calculate_summary(df)
                render_metrics(summary)

                st.info(tr("upload_info"))

                if st.button(tr("upload_cta"), type="primary"):
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
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"## {tr('details_title')}")
    st.markdown(f"<div style='color:var(--muted);'>{tr('details_subtitle')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info(tr("no_file_info"))
        if st.button(tr("button_back")):
            st.session_state.page = "main"
    else:
        if not st.session_state.ai_processed:
            batch_size = 30
            count = min(len(df), 2000)
            total_batches = math.ceil(count / batch_size)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"### {tr('ai_progress_title')}")
            status_placeholder = st.empty()
            status_placeholder.markdown(f"<div class='progress-text'>{tr('ai_progress_start')}</div>", unsafe_allow_html=True)
            progress_placeholder = st.empty()
            progress_placeholder.markdown(
                "<div class='progress-bar'><span style='width:0%'></span></div>", unsafe_allow_html=True
            )
            status_placeholder.markdown(f"<div class='progress-text'>{tr('ai_progress_sending')}</div>", unsafe_allow_html=True)

            try:
                ai_df = run_ai_categorization(
                    df, progress_placeholder=progress_placeholder, status_placeholder=status_placeholder
                )
                progress_placeholder.markdown(
                    "<div class='progress-bar'><span style='width:100%'></span></div>",
                    unsafe_allow_html=True,
                )
                status_placeholder.markdown(f"<div class='progress-text'>{tr('ai_done')}</div>", unsafe_allow_html=True)
                st.session_state.df_ai = ai_df
                st.session_state.ai_processed = True
                st.session_state.ai_error = None
            except Exception as exc:  # noqa: BLE001
                progress_placeholder.empty()
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
            st.markdown(f"## {tr('ai_overview')}")
            generate_report_cards(ai_df)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            plotly_template = "plotly_dark" if st.session_state.get("theme") == "dark" else "plotly_white"
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {tr('chart_expenses')}")
                expense_totals = prepare_category_totals(ai_df, positive=False)
                if not expense_totals.empty:
                    fig_expense = px.bar(
                        expense_totals,
                        x="Total",
                        y="Category",
                        orientation="h",
                        text="Total",
                        color="Total",
                        color_continuous_scale="Blues",
                        template=plotly_template,
                        height=420,
                    )
                    fig_expense.update_traces(marker_line_color="#0f4c75", marker_line_width=1.4, texttemplate="%{text:$,.0f}")
                    fig_expense.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_title=tr("summary_exits"),
                        yaxis_title=tr("category_label"),
                        margin=dict(l=0, r=0, t=40, b=0),
                        font=dict(size=14),
                    )
                    st.plotly_chart(fig_expense, use_container_width=True)
                else:
                    st.info(tr("no_expenses"))
            with col2:
                st.markdown(f"#### {tr('chart_income')}")
                income_totals = prepare_category_totals(ai_df, positive=True)
                if not income_totals.empty:
                    fig_income = px.bar(
                        income_totals,
                        x="Category",
                        y="Total",
                        text="Total",
                        color="Total",
                        color_continuous_scale="Teal",
                        template=plotly_template,
                        height=420,
                    )
                    fig_income.update_traces(marker_line_color="#0066ff", marker_line_width=1.4, texttemplate="%{text:$,.0f}")
                    fig_income.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_title=tr("category_label"),
                        yaxis_title=tr("summary_entries"),
                        margin=dict(l=0, r=0, t=40, b=0),
                        font=dict(size=14),
                    )
                    st.plotly_chart(fig_income, use_container_width=True)
                else:
                    st.info(tr("no_income"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"#### {tr('chart_balance')}")
            balance_df = ai_df.copy()
            balance_df["Amount"] = pd.to_numeric(balance_df["Amount"], errors="coerce")
            balance_df["Date"] = pd.to_datetime(balance_df["Date"], errors="coerce")
            balance_df = balance_df.replace([math.inf, -math.inf], pd.NA).dropna(subset=["Date", "Amount"])
            balance_df = balance_df.sort_values("Date")
            balance_df["Running Balance"] = balance_df["Amount"].cumsum()
            if not balance_df.empty:
                balance_chart = px.line(
                    balance_df,
                    x="Date",
                    y="Running Balance",
                    markers=True,
                    line_shape="spline",
                    template=plotly_template,
                    height=420,
                )
                balance_chart.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode="x unified",
                    font=dict(size=14),
                )
                st.plotly_chart(balance_chart, use_container_width=True)
            else:
                st.info(tr("file_empty"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card styled-table'>", unsafe_allow_html=True)
            st.markdown(f"#### {tr('table_full')}")
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
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"### {tr('export_section')}")
            zoho_csv, qb_csv, vendors_csv = prepare_downloads(ai_df)
            colz, colq, colv = st.columns(3)
            with colz:
                st.download_button(
                    tr("download_zoho"),
                    data=zoho_csv,
                    file_name="zoho_books_transactions.csv",
                    mime="text/csv",
                )
            with colq:
                st.download_button(
                    tr("download_qb"),
                    data=qb_csv,
                    file_name="quickbooks_transactions.csv",
                    mime="text/csv",
                )
            with colv:
                st.download_button(
                    tr("download_vendors"),
                    data=vendors_csv,
                    file_name="vendors.csv",
                    mime="text/csv",
                )

            if len(df) > 2000:
                st.warning(tr("ai_only_first_2000"))

        if st.button(tr("button_back")):
            st.session_state.page = "main"
