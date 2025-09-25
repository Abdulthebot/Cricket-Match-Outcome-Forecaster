import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IPL Match Forecaster",
    page_icon="🏏",
    layout="centered"
)

# --- LOAD SAVED ARTIFACTS ---
try:
    model = joblib.load('cricket_model.pkl')
    encoders = joblib.load('encoders.pkl')
except FileNotFoundError:
    st.error("Model or encoder files not found. Please run the `cricket_model_training.ipynb` notebook first.")
    st.stop()

# --- APP LAYOUT ---
st.title("🏏 IPL Match Outcome Forecaster")
st.markdown("Select the teams, venue, and toss details to predict the match winner.")

# Get the list of teams and venues from the encoders
teams = encoders['team1'].classes_
venues = encoders['venue'].classes_

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", options=teams, index=4) # Default to a common team
    toss_winner = st.selectbox("Select Toss Winner", options=teams, index=4)

with col2:
    team2 = st.selectbox("Select Team 2", options=teams, index=5) # Default to another common team
    toss_decision = st.selectbox("Select Toss Decision", options=['field', 'bat'], index=0)

venue = st.selectbox("Select Venue", options=venues)

# --- PREDICTION LOGIC ---
if st.button("Predict Win Probability"):
    if team1 == team2:
        st.error("Please select two different teams.")
    else:
        try:
            # Prepare the input for the model using the stored encoders
            input_data = pd.DataFrame({
                'team1': [encoders['team1'].transform([team1])[0]],
                'team2': [encoders['team2'].transform([team2])[0]],
                'venue': [encoders['venue'].transform([venue])[0]],
                'toss_winner': [encoders['toss_winner'].transform([toss_winner])[0]],
                'toss_decision': [encoders['toss_decision'].transform([toss_decision])[0]]
            })

            # Get prediction probabilities
            win_probabilities = model.predict_proba(input_data)[0]

            # Get the probability for each selected team
            prob_team1 = win_probabilities[encoders['winner'].transform([team1])[0]]
            prob_team2 = win_probabilities[encoders['winner'].transform([team2])[0]]
            
            # Normalize probabilities to sum to 100% between the two teams
            total_prob = prob_team1 + prob_team2
            if total_prob == 0:
                st.warning("Could not determine win probability. The model may not have enough data for this specific matchup.")
                st.stop()
                
            win_prob_team1 = round((prob_team1 / total_prob) * 100)
            win_prob_team2 = round((prob_team2 / total_prob) * 100)

            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label=f"**{team1} Win Probability**", value=f"{win_prob_team1}%")
                st.progress(win_prob_team1 / 100)
            with res_col2:
                st.metric(label=f"**{team2} Win Probability**", value=f"{win_prob_team2}%")
                st.progress(win_prob_team2 / 100)

        except Exception as e:
            st.error(f"An error occurred. It's possible the selected teams/venue are not in the model's training data. Error: {e}")
