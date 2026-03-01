# 📊 Corporate AI-Powered RFM & CLTV Analytics Platform

This project is an enterprise-grade web application that automates **RFM (Recency, Frequency, Monetary)** and **CLTV (Customer Lifetime Value)** analyses. It integrates a **Local AI Agent (Llama 3)** to interpret the segmentation results and provide actionable marketing strategies with 100% data privacy.

## 🚀 Features

* **Integrated Analysis Pipeline:** Computes both historical customer loyalty (RFM) and future revenue predictions (CLTV) with a single dataset upload.
* **Dynamic Column Mapping:** Flexible UI that maps required analytical columns without forcing strict naming conventions on your raw CSV data.
* **100% Data Privacy (Local LLM):** Customer data (CRM) is never sent to external servers. The AI agent utilizes LangChain and Ollama to run the Llama 3 model entirely locally via a Context-Injection architecture.
* **Autonomous CRM Consultant:** The integrated AI agent reads the segmentation matrices and answers strategic questions (e.g., "Which segment should I target for ads?") using professional business acumen.
* **Fault-Tolerant Engine:** Engineered to handle parsing errors, zero-division cases in tenure, and datatype mismatches safely.

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/rfm_cltv_project.git](https://github.com/YOUR_USERNAME/rfm_cltv_project.git)
   cd rfm_cltv_project
