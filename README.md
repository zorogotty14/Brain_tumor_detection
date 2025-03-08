# Brain Tumor Detection Web Application

## Overview
This web application is built using Flask and is designed to detect brain tumors based on user inputs such as MRI images. The application leverages a machine learning model to provide predictions, helping users to better understand their health conditions and seek timely medical advice.

## Features
- **User-friendly web interface** to upload MRI images.
- **Predicts potential brain tumor types** using trained machine learning models.
- **Provides explanations and insights** into the predictions.
- **Customizable and extendable** to support additional brain conditions and input features.

## Problem Statement
Brain tumors, including types such as glioma, meningioma, pituitary tumors, and the absence of a tumor ('notumor'), are serious medical conditions requiring prompt diagnosis and treatment. Delayed diagnosis can lead to critical health consequences. However, access to advanced diagnostic tools is often limited, particularly in remote or underserved areas.

This web application aims to address these challenges by:
- Providing an AI-powered tool for quick and preliminary detection of brain tumors.
- Enhancing accessibility for individuals without immediate access to healthcare facilities.
- Supporting healthcare professionals by streamlining initial diagnostic processes and prioritizing urgent cases.

## Dataset and Models
The dataset and pre-trained models for this project can be accessed from the following link:
[Brain Tumor Detection Dataset and Models](https://drive.google.com/drive/folders/1bTH3St3Slzp2t0F7CuP-CBNfLRrP9_nL?usp=sharing)

If access is required, please reach out via email to request permission.

## Use Case: Predict Brain Tumor Type
### 1. Preconditions
- User must be registered and logged in to access the application.
- The trained machine learning model (`brain_tumor_detection_model_with_others.h5`) must be loaded successfully.
- The LM Studio with the `Meta-Llama-3.1-8B-Instruct-GGUF` model must be running on localhost at port 1234.
- The `static/uploads` folder must exist for storing uploaded images.

### 2. Main Flow
1. **Upload Image:** User uploads an MRI image file via the provided form.
2. **Process Input:** Application preprocesses the image and generates predictions.
3. **Display Results:** Application shows the predicted tumor type and detailed information about the condition.

### 3. Alternative Flows
- **No File Uploaded:** Displays an error if no file is uploaded.
- **Unsupported File Format:** Shows an error for invalid file types.
- **Model Error:** Informs the user if predictions cannot be generated.

## Prerequisites
- Python 3.x
- Flask
- A virtual environment (recommended)
- LM Studio

## Installation
### Step 1: Clone the repository
```bash
git clone https://github.com/zorogotty14/Brain_tumor_detection.git
cd Brain_tumor_detection
```

### Step 2: Create and activate a virtual environment
```bash
# On Linux/MacOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install the dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up and Run LM Studio
- Download and install LM Studio from the [official GitHub repository](https://github.com/CySBeR-Lab/HyperDstIDFormer/tree/main).
- Ensure LM Studio is running on the default port 1234.

### Step 5: Run the Flask application
```bash
python App_Densenet.py
```
The application will be accessible at `http://127.0.0.1:3000/`.

## Usage
1. Open the application in your web browser.
2. Upload an MRI image of the brain.
3. Click on the "Predict" button to analyze the image.
4. View the predicted tumor type and related information on the results page.


## Project Structure
```
brain-tumor-detection-app/
├── App_Densenet.py                 # Main application file
├── App_VGG16.py            # Test application file
├── templates/
│   ├── index.html         # Home page template
│   ├── result.html        # Results page template
│   ├── style.css          # Stylesheet
│   └── error.html         # Error page template
├── models/
│   └── brain_tumor_detection_model_with_others.h5  # Pre-trained ML model
├── requirements.txt       # List of dependencies
└── README.md              # Readme file
```

## Customization
- **Updating the Model:** Replace the `brain_tumor_detection_model_with_others.h5` file to improve accuracy or add new tumor types.
- **Frontend Customization:** Modify HTML templates in the `templates/` directory to adjust the UI.

## Future Improvements
- Add support for more brain tumor types.
- Enhance the prediction model with newer algorithms.
- Integrate user authentication and profile management.

## Contributing
To contribute to this project, open a pull request or submit an issue on GitHub.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please reach out to Gautham Gali at [ggali14@vt.edu].

