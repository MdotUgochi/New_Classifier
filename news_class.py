import streamlit as st
import joblib

#Load the  files
model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("TfidfVectorizer.pkl")
accuracy = joblib.load("model_accuracy.pkl")

#Descriptions
st.title("News Classification App")
st.write("Enter a news headline or article below to classify it.")

# model's overall accuracy in  percentage with 2 decimal places
st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

#Text Inputbox
user_input = st.text_area("Paste text here:")

#the Predict Button
if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # run input using the loaded vectorizer
        input_vector = vectorizer.transform([user_input])
        
        #prediction
        prediction = model.predict(input_vector)
        
        # Display result
        st.success(f"Predicted Category: **{prediction[0]}**")