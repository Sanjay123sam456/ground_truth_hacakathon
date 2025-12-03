import os
import io
import base64
import json
import tempfile
import requests
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from weasyprint import HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

load_dotenv()

OPENROUTER_KEY_ENV = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
LLM_INPUT_CHAR_LIMIT = 60_000

# -----------------------------
# OpenRouter API call
# -----------------------------
def call_openrouter(api_key: str, model: str, messages: list, site_url="http://localhost:8501", site_title="Streamlit App", timeout=60) -> str:
    if not api_key:
        raise RuntimeError("No OpenRouter API key provided.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": site_url,
        "X-Title": site_title,
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 4000}
    resp = requests.post(url=url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return j.get("output", {}).get("text", "") or j.get("choices", [{}])[0].get("text", "") or ""

# -----------------------------
# Smart data chunking for LLM
# -----------------------------
def prepare_data_for_llm(df: pd.DataFrame, max_chars: int = 50000) -> str:
    """
    Intelligently prepare data for LLM using chunking strategy.
    Sends complete data in chunks if needed, prioritizing statistical summary.
    """
    # Start with comprehensive statistical summary
    summary_parts = []
    
    # 1. Basic info
    summary_parts.append(f"DATASET OVERVIEW:")
    summary_parts.append(f"- Total Rows: {len(df)}")
    summary_parts.append(f"- Total Columns: {len(df.columns)}")
    summary_parts.append(f"- Columns: {', '.join(df.columns.tolist())}\n")
    
    # 2. Numeric columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        summary_parts.append("NUMERIC COLUMNS STATISTICS:")
        desc = df[numeric_cols].describe().transpose()
        desc['median'] = df[numeric_cols].median()
        desc['mode'] = df[numeric_cols].mode().iloc[0] if len(df[numeric_cols].mode()) > 0 else None
        summary_parts.append(desc.to_string())
        summary_parts.append("")
    
    # 3. Categorical columns distribution
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        summary_parts.append("CATEGORICAL COLUMNS DISTRIBUTION:")
        for col in cat_cols:
            value_counts = df[col].value_counts().head(10)
            summary_parts.append(f"\n{col}:")
            summary_parts.append(value_counts.to_string())
        summary_parts.append("")
    
    # 4. Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append("MISSING VALUES:")
        summary_parts.append(missing[missing > 0].to_string())
        summary_parts.append("")
    
    # 5. Correlation matrix for numeric columns
    if len(numeric_cols) >= 2:
        summary_parts.append("CORRELATION MATRIX:")
        corr = df[numeric_cols].corr()
        summary_parts.append(corr.to_string())
        summary_parts.append("")
    
    statistical_summary = "\n".join(summary_parts)
    
    # Now add actual data rows
    # Calculate how much space we have left
    remaining_chars = max_chars - len(statistical_summary) - 500  # buffer
    
    if remaining_chars > 1000:
        # Try to fit as many rows as possible
        csv_text = df.to_csv(index=False)
        
        if len(csv_text) <= remaining_chars:
            # All data fits
            data_section = f"\n\nCOMPLETE DATASET:\n{csv_text}"
        else:
            # Sample strategically: head + tail + random middle
            n_rows = len(df)
            head_rows = min(50, n_rows // 3)
            tail_rows = min(50, n_rows // 3)
            middle_rows = min(100, n_rows - head_rows - tail_rows)
            
            parts = []
            parts.append(f"\n\nDATA SAMPLE (showing {head_rows + tail_rows + middle_rows} of {n_rows} rows):")
            parts.append("\nFIRST ROWS:")
            parts.append(df.head(head_rows).to_csv(index=False))
            
            if middle_rows > 0:
                middle_start = head_rows
                middle_end = head_rows + middle_rows
                parts.append("\nMIDDLE ROWS:")
                parts.append(df.iloc[middle_start:middle_end].to_csv(index=False))
            
            parts.append("\nLAST ROWS:")
            parts.append(df.tail(tail_rows).to_csv(index=False))
            
            data_section = "".join(parts)
            
            # Trim if still too long
            if len(statistical_summary + data_section) > max_chars:
                data_section = data_section[:remaining_chars]
    else:
        data_section = f"\n\nDATA SAMPLE (first 20 rows):\n{df.head(20).to_csv(index=False)}"
    
    final_text = statistical_summary + data_section
    return final_text[:max_chars]

# -----------------------------
# Enhanced chart generation
# -----------------------------
def create_enhanced_charts(df: pd.DataFrame):
    chart_paths = []
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # 1. Distribution charts for numeric columns
    for col in numeric_cols[:4]:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1e3a8a", edgecolor="white", linewidth=0.5)
        ax.set_title(f"{col} Distribution", fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout(pad=0.5)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
        tmpfile.close()
        fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        chart_paths.append(tmpfile.name)

    # 2. Top categories bar chart
    for col in cat_cols[:3]:
        counts = df[col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = sns.barplot(y=counts.index.astype(str), x=counts.values, ax=ax, palette="Blues_r")
        ax.set_title(f"Top {col} (Count)", fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("Count", fontsize=8)
        ax.set_ylabel(col, fontsize=8)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=2, fontsize=7)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout(pad=0.5)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
        tmpfile.close()
        fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        chart_paths.append(tmpfile.name)

    # 3. Correlation heatmap (if multiple numeric columns)
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
        ax.set_title("Correlation Matrix", fontsize=10, fontweight='bold', pad=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=0.5)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
        tmpfile.close()
        fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        chart_paths.append(tmpfile.name)

    # 4. Box plots for numeric columns (showing outliers)
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        df_melted = df[numeric_cols[:4]].melt(var_name='Variable', value_name='Value')
        sns.boxplot(data=df_melted, x='Variable', y='Value', ax=ax, palette='Set2')
        ax.set_title("Distribution & Outliers", fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=0.5)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
        tmpfile.close()
        fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        chart_paths.append(tmpfile.name)

    return chart_paths

# -----------------------------
# Enhanced PDF generation with modern styling
# -----------------------------
def create_pdf_weasyprint(title: str, subtitle: str, markdown_text: str, kpis: dict = None, chart_paths: list = None) -> bytes:
    css = """
    <style>
      @page {
        size: A4;
        margin: 12mm 10mm;
      }
      
      body {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: #1f2937;
        line-height: 1.4;
        margin: 0;
        padding: 0;
      }
      
      .header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 20px 20px;
        margin: -12mm -10mm 15px -10mm;
        border-radius: 0 0 8px 8px;
      }
      
      h1 {
        margin: 0 0 8px 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
      }
      
      h2 {
        margin: 0;
        font-size: 14px;
        font-weight: 400;
        opacity: 0.95;
      }
      
      .kpi-container {
        display: flex;
        justify-content: space-between;
        margin: 15px 0 20px 0;
        flex-wrap: wrap;
        gap: 8px;
      }
      
      .kpi {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 12px 15px;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
      }
      
      .kpi-label {
        font-size: 11px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
      }
      
      .kpi-value {
        font-size: 24px;
        color: #1e3a8a;
        font-weight: 700;
      }
      
      .section {
        margin-bottom: 18px;
        page-break-inside: avoid;
      }
      
      .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #3b82f6;
      }
      
      .insights-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 15px;
      }
      
      .insight-card {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 10px 12px;
        border-radius: 4px;
      }
      
      .insight-card h3 {
        font-size: 11px;
        color: #1e3a8a;
        margin: 0 0 6px 0;
        font-weight: 600;
      }
      
      ul {
        margin: 8px 0;
        padding-left: 18px;
      }
      
      li {
        font-size: 10px;
        margin-bottom: 6px;
        line-height: 1.5;
        color: #374151;
      }
      
      li::marker {
        color: #3b82f6;
        font-weight: bold;
      }
      
      .charts-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 12px 0;
        width: 100%;
      }
      
      .chart-container {
        text-align: center;
        page-break-inside: avoid;
        width: 100%;
      }
      
      img {
        width: 100%;
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        display: block;
      }
      
      .footer {
        font-size: 8px;
        color: #94a3b8;
        text-align: center;
        margin-top: 15px;
        padding-top: 8px;
        border-top: 1px solid #e2e8f0;
      }
      
      .page-break {
        page-break-before: always;
        margin-top: 0;
      }
      
      .recommendations {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        border-radius: 4px;
        margin-top: 12px;
        margin-bottom: 8px;
      }
      
      .recommendations h3 {
        color: #92400e;
        font-size: 12px;
        margin: 0 0 8px 0;
        font-weight: 700;
      }
    </style>
    """
    
    # Parse markdown sections
    sections = {
        'metrics': [],
        'insights': [],
        'trends': [],
        'recommendations': []
    }
    
    current_section = 'insights'
    for line in markdown_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if 'metric' in lower_line or 'kpi' in lower_line:
            current_section = 'metrics'
            continue
        elif 'insight' in lower_line or 'finding' in lower_line:
            current_section = 'insights'
            continue
        elif 'trend' in lower_line or 'pattern' in lower_line:
            current_section = 'trends'
            continue
        elif 'recommend' in lower_line or 'action' in lower_line:
            current_section = 'recommendations'
            continue
        
        # Remove markdown symbols
        clean_line = line.lstrip('-*•#').strip()
        if clean_line:
            sections[current_section].append(clean_line)
    
    # Build KPIs HTML
    kpi_html = ""
    if kpis:
        kpi_html = "<div class='kpi-container'>"
        for k, v in kpis.items():
            kpi_html += f"""
            <div class='kpi'>
                <div class='kpi-label'>{k}</div>
                <div class='kpi-value'>{v}</div>
            </div>
            """
        kpi_html += "</div>"
    
    # Build insights cards
    insights_html = ""
    if sections['insights'] or sections['trends']:
        insights_html = "<div class='insights-grid'>"
        
        if sections['insights']:
            insights_html += "<div class='insight-card'><h3>📊 Key Insights</h3><ul>"
            for item in sections['insights'][:5]:
                insights_html += f"<li>{item}</li>"
            insights_html += "</ul></div>"
        
        if sections['trends']:
            insights_html += "<div class='insight-card'><h3>📈 Trends & Patterns</h3><ul>"
            for item in sections['trends'][:5]:
                insights_html += f"<li>{item}</li>"
            insights_html += "</ul></div>"
        
        insights_html += "</div>"
    
    # Build charts grid with proper file paths
    charts_html = ""
    if chart_paths:
        charts_html = "<div class='section'><div class='section-title'>Visual Analysis</div><div class='charts-grid'>"
        for i, path in enumerate(chart_paths):
            # Ensure proper file URL format
            file_url = path.replace('\\', '/')
            if not file_url.startswith('file://'):
                file_url = f'file:///{file_url}' if file_url[1] == ':' else f'file://{file_url}'
            charts_html += f"<div class='chart-container'><img src='{file_url}' alt='Chart {i+1}'></div>"
        charts_html += "</div></div>"
    
    # Build recommendations
    rec_html = ""
    if sections['recommendations']:
        rec_html = "<div class='recommendations'><h3>💡 Strategic Recommendations</h3><ul>"
        for item in sections['recommendations'][:6]:
            rec_html += f"<li>{item}</li>"
        rec_html += "</ul></div>"
    
    # Additional metrics
    additional_bullets = ""
    if sections['metrics']:
        additional_bullets = "<div class='section'><div class='section-title'>Additional Metrics</div><ul>"
        for item in sections['metrics'][:8]:
            additional_bullets += f"<li>{item}</li>"
        additional_bullets += "</ul></div>"
    
    final_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        {css}
    </head>
    <body>
        <div class='header'>
            <h1>{title}</h1>
            <h2>{subtitle}</h2>
        </div>
        
        {kpi_html}
        {insights_html}
        {additional_bullets}
        
        {charts_html}
        {rec_html}
        
        <div class='footer'>Generated with OpenRouter AI Analytics | Confidential Executive Report</div>
    </body>
    </html>
    """
    
    pdf_bytes = HTML(string=final_html).write_pdf()
    return pdf_bytes

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Executive Report Generator", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;}
    </style>
    <div class='main-header'>📊 Executive Report Generator</div>
    <div class='sub-header'>Transform your data into professional executive reports with AI-powered insights</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
user_api_key = st.sidebar.text_input("OpenRouter API Key", value=OPENROUTER_KEY_ENV, type="password")
model_name = st.sidebar.text_input("Model", value=DEFAULT_OPENROUTER_MODEL)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use filters to focus your analysis on specific data segments")

# Load data
st.header("📁 Step 1: Load Your Data")
col1, col2 = st.columns([2,1])
df = None

with col1:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    raw_text = st.text_area("Or paste CSV data", height=140, placeholder="Paste your CSV data here...")

with col2:
    st.write("**Quick Start:**")
    if st.button("📊 Load Sample Dataset", use_container_width=True):
        df = pd.DataFrame({
            "employee": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
            "department": ["Sales", "Sales", "Engineering", "Marketing", "Engineering", "Sales", "Marketing", "Engineering"],
            "revenue": [88000, 92000, 70000, 95000, 80000, 110000, 85000, 75000],
            "satisfaction": [4.2, 4.5, 3.8, 4.7, 4.1, 4.8, 4.3, 3.9],
            "years": [3, 2, 5, 4, 6, 8, 3, 4]
        })
        st.success("✅ Sample data loaded!")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif raw_text and df is None:
    df = pd.read_csv(io.StringIO(raw_text))

if df is None:
    st.info("👆 Please upload a CSV file, paste data, or load the sample dataset to begin")
    st.stop()

st.success(f"✅ Data loaded: **{len(df)} rows** × **{len(df.columns)} columns**")
with st.expander("👀 Preview data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# Filtering
st.header("🔍 Step 2: Filter Data (Optional)")
cols = df.columns.tolist()
filter_col = st.selectbox("Select column to filter", ["(none)"] + cols)
filtered_df = df.copy()

if filter_col != "(none)":
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        operator = st.selectbox("Operator", ["equals", "contains", ">", "<", ">=", "<="])
    with col2:
        filter_value = st.text_input("Value")
    with col3:
        if st.button("🔎 Apply Filter", use_container_width=True):
            try:
                if operator == "equals":
                    filtered_df = df[df[filter_col].astype(str) == filter_value]
                elif operator == "contains":
                    filtered_df = df[df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)]
                else:
                    val = float(filter_value)
                    numeric_col = pd.to_numeric(df[filter_col], errors='coerce')
                    temp_df = df.copy()
                    temp_df["_num"] = numeric_col
                    if operator == ">": filtered_df = temp_df[temp_df["_num"] > val]
                    elif operator == "<": filtered_df = temp_df[temp_df["_num"] < val]
                    elif operator == ">=": filtered_df = temp_df[temp_df["_num"] >= val]
                    elif operator == "<=": filtered_df = temp_df[temp_df["_num"] <= val]
                    filtered_df = filtered_df.drop(columns=["_num"])
                st.success(f"✅ Filter applied: **{len(filtered_df)}** rows remaining")
            except Exception as e:
                st.error(f"❌ Filter error: {e}")
                filtered_df = df.copy()

if len(filtered_df) != len(df):
    st.info(f"📊 Filtered: **{len(filtered_df)}** of **{len(df)}** rows")

with st.expander("👀 View filtered data", expanded=False):
    st.dataframe(filtered_df.head(50), use_container_width=True)

# Generate report
st.header("🚀 Step 3: Generate Executive Report")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("📋 Report Title", value="Executive Data Analysis Report")
with col2:
    subtitle = st.text_input("📅 Subtitle", value=f"Analysis Date: {pd.Timestamp.now().strftime('%B %d, %Y')}")

default_system = (
    "You are a senior business intelligence analyst creating an executive report. "
    "Produce concise, actionable insights in bullet-point format. Structure your response with clear sections: "
    "Key Metrics, Key Insights, Trends & Patterns, and Strategic Recommendations. "
    "Each bullet should be one clear sentence. Focus on business impact and actionable intelligence."
)

default_user_prompt = (
    "Analyze this dataset comprehensively and provide:\n\n"
    "KEY METRICS (5-8 bullets):\n"
    "- Critical statistics and benchmarks\n"
    "- Performance indicators\n\n"
    "KEY INSIGHTS (5-7 bullets):\n"
    "- Most important discoveries\n"
    "- Notable anomalies or standout data points\n\n"
    "TRENDS & PATTERNS (4-6 bullets):\n"
    "- Observable correlations\n"
    "- Temporal or categorical patterns\n\n"
    "STRATEGIC RECOMMENDATIONS (4-6 bullets):\n"
    "- Actionable next steps\n"
    "- Risk mitigation strategies\n"
    "- Opportunity areas\n\n"
    "Keep all points concise and executive-focused."
)

with st.expander("⚙️ Advanced Settings", expanded=False):
    system_msg = st.text_area("System message", value=default_system, height=120)
    user_prompt = st.text_area("Analysis prompt", value=default_user_prompt, height=200)
    max_chars = st.slider("Maximum data size for AI (characters)", 10000, 60000, 50000, 5000)
    st.info("💡 Using smart chunking - complete dataset statistics will be included")

generate = st.button("🎯 Generate Executive Report", type="primary", use_container_width=True)

if generate:
    with st.spinner("🔄 Analyzing data and generating visualizations..."):
        # Calculate KPIs
        numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        kpis = {}
        for col in numeric_cols[:4]:
            avg_val = filtered_df[col].mean()
            if avg_val > 1000:
                kpis[col] = f"{avg_val:,.0f}"
            else:
                kpis[col] = f"{avg_val:.2f}"
        
        # Generate charts
        chart_paths = create_enhanced_charts(filtered_df)
        
        # Prepare LLM input with smart chunking
        data_text = prepare_data_for_llm(filtered_df, max_chars=max_chars)
        full_user_content = (user_prompt + "\n\n" + data_text)[:LLM_INPUT_CHAR_LIMIT]
        
        api_key_to_use = user_api_key or OPENROUTER_KEY_ENV
        if not api_key_to_use:
            st.error("❌ Please provide an OpenRouter API key in the sidebar")
            st.stop()
        
        try:
            with st.spinner("🤖 AI is analyzing your data..."):
                assistant_text = call_openrouter(
                    api_key=api_key_to_use,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": full_user_content},
                    ],
                )
        except Exception as e:
            st.error(f"❌ AI analysis failed: {e}")
            st.stop()
        
        # Generate PDF
        try:
            with st.spinner("📄 Creating PDF report..."):
                pdf_bytes = create_pdf_weasyprint(
                    title=title,
                    subtitle=subtitle,
                    markdown_text=assistant_text,
                    kpis=kpis,
                    chart_paths=chart_paths
                )
        except Exception as e:
            st.error(f"❌ PDF generation failed: {e}")
            st.stop()
        
        st.success("✅ Executive report generated successfully!")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"executive_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        # PDF preview
        st.markdown("### 📄 Report Preview")
        try:
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception:
            st.info("Preview unavailable in this browser. Please download the PDF to view.")
        
        # Show AI insights
        with st.expander("🤖 View AI-Generated Insights", expanded=False):
            st.markdown(assistant_text)