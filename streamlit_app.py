import streamlit as st
import pickle

# Load resources
with open('model.pkl', 'rb') as m_file:
    
    model = pickle.load(m_file)

with open('vectorizer.pkl', 'rb') as v_file:
    vectorizer = pickle.load(v_file)

# UI Elements
st.title("📬 AI Spam Detector")
st.write("Enter an SMS or Email text below to see if it's classified as Spam or Ham.")

# Single-page form input
user_input = st.text_area("Message to analyze:", placeholder="Type your text here...", height=150)

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform text and predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        
        # Display corresponding output banner
        if prediction == 1:
            st.error("🚨 This looks like SPAM.")
        else:
            st.success("✅ This looks like a clean HAM message.")
    else:
        st.warning("Please type a message before pressing Predict.")