# Team-BruteForce4O4-H2S
#  FairHire: Algorithmic Bias Auditor & Mitigation Engine
**Developed by Team BruteForce4O4 | Google Solution Challenge 2026**

FairHire is a proactive tool designed to solve the "Black Box" problem in AI-driven recruitment. Many automated hiring systems unintentionally learn and amplify historical human biases. FairHire audits these datasets, quantifies the disparity, and provides an actionable "Fairness Offset" to restore equity in real-time.

---

## The Core Problem
AI models trained on historical data often inherit a "Bias Tax." In our testing phase, we identified a **15.54 point gender gap** where qualified female candidates were systematically underscored by the algorithm. FairHire bridges this gap.

## Key Features
- **Statistical Audit:** Instant calculation of Gender and Age bias metrics using Python & Pandas.
- **AI-Powered Analysis:** Integration with **Google Gemini 2.5 Flash-Lite** to generate high-level executive summaries and legal risk assessments.
- **Bias Mitigation Engine:** A one-click toggle that applies a mathematical offset to equalize scores for disadvantaged groups based on detected gaps.
- **Transparency Dashboard:** A clean Streamlit interface showing side-by-side leaderboards (Original vs. Mitigated).

## Tech Stack
- **AI Engine:** Google Gemini 2.5 Flash-Lite (via Google Generative AI SDK)
- **Interface:** Streamlit (Cloud Deployed)
- **Data Processing:** Python 3.13, Pandas
- **Visualization:** Matplotlib, Seaborn

## System Architecture
The data flows through a four-stage pipeline to ensure transparency:
1. **Input:** CSV dataset ingestion.
2. **Audit Layer:** Mathematical bias detection using Pandas.
3. **Interpretation Layer:** Gemini AI converts raw stats into ethical memos.
4. **Output Dashboard:** Real-time visualization for HR decision-makers.

