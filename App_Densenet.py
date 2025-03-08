from flask import Flask, render_template, request, jsonify, redirect, url_for,session
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from openai import OpenAI
import re

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)


# Define the model path and load the model
MODEL_PATH = "./models/Densenet_Final.h5"
model = load_model(MODEL_PATH)

# Define confidence threshold for classifying as "not a brain MRI"
CONFIDENCE_THRESHOLD = 0.6

# Define the classes based on your model training
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to get classification details based on predicted class
def get_tumor_info(tumor_type):
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=[
        {"role": "system", "content": "You are a knowledgeable healthcare assistant specializing in neurology and oncology."},
        {"role": "user", "content": f"Could you provide an in-depth analysis of {tumor_type} brain tumors, covering their classification, clinical presentation, available treatment modalities, and expected outcomes?"}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


def format_tumor_info(text):
    # Wrap main headings (e.g., "Types of Meningiomas") as <h3>
    text = re.sub(r"\*\*(.+?)\*\*", r"<h3 class='section-heading'>\1</h3>", text)
    
    # Wrap bolded items (e.g., subheadings or important terms) as <strong>
    text = re.sub(r"(\*\*.+?\*\*)", lambda match: f"<strong>{match.group(0)[2:-2]}</strong>", text)
    
    # Replace numbered or bullet lists with <li> items
    text = re.sub(r"(?:•\s|\d+\.\s)(.+)", r"<li>\1</li>", text)
    
    # Wrap lists with <ul> tags
    text = re.sub(r"(</h3>)\s*((?:<li>.+?</li>\s*)+)", r"\1<ul class='no-marker'>\2</ul>", text)
    
    # Replace new lines with paragraph tags
    text = re.sub(r"\n+", r"</p><p>", text)
    text = f"<p>{text}</p>"

    return text



# Route for handling chatbot messages with session history
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    # Retrieve conversation history from session or initialize with tumor info
    conversation_history = session.get('conversation_history', [])
    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=conversation_history,
        temperature=0.7,
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    # Update session with the conversation history
    session['conversation_history'] = conversation_history
    return jsonify({"reply": reply})

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Route to handle prediction and store initial conversation context
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    upload_folder = 'C:/Users/gagan/OneDrive/Desktop/Projects/AIT736_Group_8_FinalProject/static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class_name = class_names[predicted_class_index]

        # Check for "No Tumor" case and adjust message
    if predicted_class_name == "No Tumor":
        result_text = "No brain tumor detected."
        tumor_info = "<p>The uploaded MRI scan shows no signs of a brain tumor.<p>"
    else:
        # Get detailed information from GPT based on predicted class
        result_text = f"Positive for Brain Tumor: {predicted_class_name}"
        raw_tumor_info = get_tumor_info(predicted_class_name)
        tumor_info = format_tumor_info(raw_tumor_info)
        # print(tumor_info)

        # Initialize conversation history with tumor information
        session['conversation_history'] = [
            {"role": "assistant", "content": f"{result_text}. {tumor_info}"}
        ]

    # Render the result template with the prediction result and GPT information
    return render_template('result.html', 
                           result_text=result_text,
                           tumor_info=tumor_info,
                           file_path=f'uploads/{file.filename}')

# Run the app
if __name__ == '__main__':
    app.run(port=3000, debug=True)