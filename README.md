# Food-Health-Predictor- Food Label Classifier with Flask

Food-Health-Predictor is a machine learning-powered web app that classifies packaged food products based on labeled nutritional and ingredient data. It uses a **supervised ML model** trained on the OpenFoodFacts dataset and served through a **Flask backend**. The app includes static styling and templated HTML for a clean UI.

## Features

- **Supervised ML Model**: Classifies food items using a trained model (`openfoodfactr.pkl`)
- **Flask Backend**: Handles prediction logic and routes
- **Static Styling**: CSS files located in the `static/` folder
- **HTML Templates**: Stored in the `templates/` folder for dynamic rendering
- **Lightweight**: Easy to deploy, simple to extend

## Project Structure
food-health-predictor/
├── app.py # Flask backend
├── openfoodfactr.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── templates/ # HTML templates
│ └── index.html
├── static/ # CSS stylesheets
│ └── style.css
├── .gitignore
└── README.md

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash

git clone https://github.com/yourusername/openfoodfactr.git
cd openfoodfactr
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py

The app will run on http://127.0.0.1:5000

### 2. Model Info
The ML model was trained on labeled data from OpenFoodFacts, targeting classification tasks such as:

Predicting food categories

Identifying unhealthy ingredients

Assessing nutrition scores

The trained model is stored as: openfoodfactr.pkl

### 3.Requirements
Key packages used (see full list in requirements.txt):

Flask

scikit-learn

pandas

numpy

joblib
