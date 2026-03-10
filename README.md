🚀 *InsightBot: The Automated CSV-to-PDF Insight Engine*  
*Tagline:* Transform raw CSV data into executive-ready PDF reports with AI-driven insights — fully automated via Streamlit in under 30 seconds.  

---

## 1. The Problem (Real World Scenario)  
*📌 Context:* In analytics workflows, account managers and analysts spend hours manually filtering CSVs, generating insights, and formatting reports for clients. This is repetitive, slow, and error-prone.  

*⚠ The Pain Point:* Manual report generation delays critical decisions. Anomalies like traffic drops or campaign performance dips may go unnoticed for days.  

*💡 My Solution:* AutoReport automates the entire pipeline. Users upload a CSV or paste raw data, apply optional filters, and within seconds, receive a *fully formatted PDF report* highlighting key trends, anomalies, and AI-generated insights.  

---

## 2. Expected End Result  
*For the User:*  

*Input:* Upload CSV file or paste raw data in the Streamlit app.  

*Action:* Summarize the data and generate a report using LLM  

*Output:* A polished PDF report containing:  

- 📊 Filtered tables and summary  
- 🤖 AI-generated narrative insights  
- ⚡ Highlighted anomalies or trends (e.g., “Clicks dropped 30% in Chicago last week”)  
- 🎨 Styled charts and tables for executive readability  

---

## 3. Technical Approach  
I aimed to build a *production-ready, end-to-end pipeline*.  

*System Architecture:*  

- *📂 Ingestion:* Upload CSVs or paste data; app loads it into a DataFrame (Pandas ).  
- *🔍 Filtering:* Users filter by columns or conditions.  
- *📝 Data Summary:* Filtered data converted into short text summary for AI context.  
- *🤖 Generative AI:*  
  - System + user prompt built with filtered data  
  - Sent to *Google Gemini LLM* via google-generativeai SDK  
  - Returns insights in *Markdown format*  
- *📄 Reporting:*  
  - Markdown → HTML (markdown2)  
  - HTML styled with professional CSS template  
  - *WeasyPrint* generates the final PDF  
- *🌐 Frontend:* Streamlit displays PDF preview and provides download button  

---

## 4. Tech Stack  

| Layer                  | Technology                       |
|------------------------|---------------------------------|
| 🐍 Language            | Above Python 3.1                |
| 📊 Data Engine          | Pandas                         |
| 🤖 AI Model            | Google Gemini (via google-generativeai) |
| 📝 Markdown → HTML      | markdown2                        |
| 🖨 PDF Generation       | WeasyPrint                       |
| 🌐 Web Interface        | Streamlit                        |

---

## 5. Challenges & Learnings  

*Challenge 1: AI Hallucinations*  
- *Issue:* LLM invented insights not present in the data.  
- *Solution:* Strict system prompt; only filtered DataFrame context passed to AI.  

*Challenge 2: Converting LLM response to html*  
- *Issue*: LLM was returning response in markdown 
- *Solution*: converted the llm markdown response to html using markdown2 and the html to pdf using weasyprint

*Challenge 3: PDF Styling*  
- *Issue:* Misaligned tables/charts.  
- *Solution:* Consistent CSS template ensures professional, executive-ready PDF.  

---

## 6. Visual Proof  

- Streamlit interface: CSV upload + filters selection  
- Preview PDF: AI-generated insights & tables  
- Downloaded PDF: Clean, styled for executives  

---

## 7. How to Run    
git clone https://github.com/Sanjay123sam456/ground_truth_hacakathon.git  
cd ground_truth_hacakathon  
pip3 install -r requirements.txt  
streamlit run datainsight.py  

