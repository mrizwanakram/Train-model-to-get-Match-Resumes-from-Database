import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('models/resume_model.keras')

# Initialize the Tokenizer and fit it on your training data
tokenizer = Tokenizer()

# Load the original training dataset to fit the tokenizer
original_training_data = pd.read_csv('data/cleaned_resume_dataset.csv')
tokenizer.fit_on_texts(original_training_data['pdf_name'])  # Assuming 'pdf_name' contains the text data

# Initialize Label Encoders
label_encoder_skills = LabelEncoder()
label_encoder_experience = LabelEncoder()

# Fit the Label Encoders
label_encoder_skills.fit(original_training_data['skills'])
label_encoder_experience.fit(original_training_data['experience'])

# Function to preprocess the resume text for the model
def preprocess_for_model(resume_text):
    resume_seq = tokenizer.texts_to_sequences([resume_text])
    padded_seq = pad_sequences(resume_seq, maxlen=100)  # Adjust maxlen based on your model
    return padded_seq

# Function to predict skills and experience using the trained model
def predict_skills_and_experience(resume_text):
    resume_padded = preprocess_for_model(resume_text)
    
    try:
        # Use the model to predict
        skills_pred, experience_pred = model.predict(resume_padded)
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging line
        return None, None

    # Get the predicted classes
    predicted_skills = np.argmax(skills_pred, axis=1)[0]
    predicted_experience = np.argmax(experience_pred, axis=1)[0]

    # Decode the predictions to get actual skill and experience values
    decoded_skill = label_encoder_skills.inverse_transform([predicted_skills])[0]
    decoded_experience = label_encoder_experience.inverse_transform([predicted_experience])[0]

    return decoded_skill, decoded_experience

# Function to preprocess job descriptions
def preprocess_text(text):
    return text.lower()  # Basic preprocessing

# Function to match resumes with the provided job description
def match_resumes_with_job_description(job_description, resumes_df):
    job_description_cleaned = preprocess_text(job_description)
    job_description_words = job_description_cleaned.split()
    
    match_results = []
    
    for index, row in resumes_df.iterrows():
        resume_name = row['pdf_name']
        resume_text = f"{row['skills']} {row['experience']}"  # Combine skills and experience
        
        # Predict skills and experience using the model
        predicted_skill, predicted_experience = predict_skills_and_experience(resume_text)
        
        if predicted_skill is None or predicted_experience is None:
            print(f"Skipping {resume_name} due to prediction error.")  # Debugging line
            continue
        
        # Check for exact match for single words
        if len(job_description_words) == 1:
            single_word = job_description_words[0]
            if single_word in predicted_skill.lower() or single_word in predicted_experience.lower():
                match_results.append({
                    "resume": resume_name,
                    "match_percentage": 100,
                })
        
        # Fuzzy matching for multiple words
        else:
            match_percentage = fuzz.token_set_ratio(job_description_cleaned, resume_text)
            match_results.append({
                "resume": resume_name,
                "match_percentage": match_percentage,
            })
    
    # Sort results by match percentage and return top 5
    sorted_results = sorted(match_results, key=lambda x: x["match_percentage"], reverse=True)[:5]
    return sorted_results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    job_description = request.form['job_description']
    data = pd.read_csv('data/cleaned_resume_dataset.csv')  # Load the dataset
    top_matching_resumes = match_resumes_with_job_description(job_description, data)

    return jsonify(top_matching_resumes)

if __name__ == '__main__':
    app.run(debug=True)
