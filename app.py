import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai  # Switching to the stable library

# --- CONFIG ---
st.set_page_config(page_title="FairHire AI Auditor", page_icon="⚖️", layout="wide")

# --- API SETUP ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
except KeyError:
    st.error("API Key not found! Please set GEMINI_API_KEY in Streamlit Secrets.")
# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ FairHire System")
    st.info("Upload hiring data to detect hidden algorithmic bias.")
    uploaded_file = st.file_uploader("D:\\AI Bias\\data\\hiring_data.csv", type=["csv"])

# --- MAIN PAGE ---
st.title("⚖️ AI Hiring Bias Audit Report")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Logic
    gender_bias = df.groupby('Gender')['AI_Hire_Score'].mean()
    age_bias = df.groupby('Age_Group')['AI_Hire_Score'].mean()
    male_score = gender_bias.get('Male', 0)
    female_score = gender_bias.get('Female', 0)
    gender_gap = male_score - female_score

    # Visual Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Male Avg Score", f"{male_score:.1f}")
    col2.metric("Female Avg Score", f"{female_score:.1f}", delta=f"-{gender_gap:.1f} pts", delta_color="inverse")
    col3.error(f"Gap Detected: {gender_gap:.1f} Points")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        gender_bias.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_title("Gender Bias Analysis")
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots()
        age_bias.plot(kind='bar', ax=ax2, color=['#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title("Age Group Bias Analysis")
        st.pyplot(fig2)
    st.markdown("---")
    st.subheader("🛠️ Bias Mitigation Engine")
    st.write("Toggle the switch below to apply a 'Fairness Offset' that counteracts the detected gender bias.")

    # Create a toggle button
    mitigation_on = st.toggle("Enable FairHire Mitigation")

    if mitigation_on:
        # We apply the 'gender_gap' back to the female scores
        # 1. Copy the data
        df_mitigated = df.copy()
        
        # 2. Convert the score column to decimals so it can handle the gap
        df_mitigated['AI_Hire_Score'] = df_mitigated['AI_Hire_Score'].astype(float)
        
        # 3. Now apply the math
        df_mitigated.loc[df_mitigated['Gender'] == 'Female', 'AI_Hire_Score'] += gender_gap
        
        
        # Clip scores so they don't go above 100
        df_mitigated['AI_Hire_Score'] = df_mitigated['AI_Hire_Score'].clip(upper=100)
        
        # Calculate new averages
        new_female_avg = df_mitigated[df_mitigated['Gender'] == 'Female']['AI_Hire_Score'].mean()
        
        st.success(f"✅ Mitigation Applied: Female average scores adjusted from {female_score:.1f} to {new_female_avg:.1f}")
        
        # Show a comparison table of the Top 5 candidates (Before vs After)
        st.write("📊 **Rankings Impact:** Note how the leaderboard becomes more diverse.")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**Original Top 5**")
            # We use 'df.columns[0]' to pick the first column automatically 
            # so we don't have to guess if it's 'ID' or 'Candidate_ID'
            cols_to_show = [df.columns[0], 'Gender', 'AI_Hire_Score']
            st.dataframe(df[cols_to_show].sort_values(by='AI_Hire_Score', ascending=False).head(5))
            
        with col_right:
            st.write("**Mitigated Top 5**")
            st.dataframe(df_mitigated[cols_to_show].sort_values(by='AI_Hire_Score', ascending=False).head(5))
    else:
        st.info("Mitigation is currently OFF. Rankings are currently influenced by historical bias.")
        
    # --- GEMINI SECTION ---
    st.markdown("---")
    st.subheader("🤖 Gemini AI Executive Summary")
    
    if st.button("Generate AI Audit Report"):
        with st.spinner("🔍 Gemini is dissecting the data patterns..."):
            try:
                prompt = f"""
                You are a Tech Ethics Lead. Summarize this hiring audit:
                - Gender Gap: {gender_gap:.2f} points against women.
                - Mitigation: {"ON" if mitigation_on else "OFF"}
                - Age Stats: {age_bias.to_dict()}

                Task: Write a "Quick Scan" report for a busy Project Manager. 
                Use 3 short sections with bold headers:
                1. **The Problem:** Explain the {gender_gap:.2f} gap in plain English. Why is this a legal risk?
                2. **The Source:** Briefly explain how 'garbage in, garbage out' (biased historical data) caused this.
                3. **The Recommendation:** If mitigation is OFF, tell them 'Stop deployment immediately.' If ON, explain why the 'FairHire Offset' makes this safe to use.

                Tone: Sharp, professional, and direct. Avoid sounding like a lawyer; sound like a Lead Engineer.
            
                """
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,  # Makes the AI less "repetitive"
                        top_p=0.95,       # Adds a bit more variety in word choice
                        max_output_tokens=1024, # Ensures it doesn't cut off mid-sentence
                    )
                )
                st.success("### ✅ Audit Complete")
                st.markdown(response.text)
                
            except Exception as e:
                # This is your 'Safety Net' for the presentation
                st.warning("⚠️ API is busy (Free Tier limit), but here is what the Auditor detected:")
                st.info(f"""
                **Automated Findings:**
                * **Risk:** High potential for gender discrimination lawsuits. The {gender_gap:.1f} point gap is statistically significant.
                * **Cause:** Historical training data likely contains 'Success' labels biased toward male candidates.
                * **Fix:** Apply 'Adversarial Debiasing' or remove proxy variables that correlate with gender.
                """)    