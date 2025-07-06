from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import requests

app = Flask(__name__)

# Load model and preprocessors
with open("openfoodfactr_new_modelnew.pkl", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle['model']
scaler = model_bundle['scaler']
imputer = model_bundle['imputer']
label_encoder = model_bundle['label_encoder']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    barcode = request.form['barcode']
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        product = data.get('product', {})
        nutriments = product.get('nutriments', {})
        product_name = product.get('product_name', 'Not available')
        brand_name = product.get('brands', 'Not available')
        image_url = product.get('image_url', 'Not available')
        claims = product.get('labels_tags', [])
        claims = ", ".join(claims) if claims else 'No health claims available.'
        ingredients_text = product.get('ingredients_text', 'Not available')

        # Check if product name is available and nutriments are present
        if product_name == 'Not available' or not nutriments:
            return render_template('index.html', error='Insufficient product data to make prediction.')

        # Create input data for prediction
        nutri_data = {
            'nutri_energy_num': [nutriments.get('energy_100g', 0)],
            'nutri_satuFat_num': [nutriments.get('saturated-fat_100g', 0)],
            'nutri_carbohydrate_num': [nutriments.get('carbohydrates_100g', 0)],
            'nutri_sugar_num': [nutriments.get('sugars_100g', 0)],
            'nutri_fiber_num': [nutriments.get('fiber_100g', 0)],
            'nutri_protein_num': [nutriments.get('proteins_100g', 0)],
            'nutri_salt_num': [nutriments.get('salt_100g', 0)],
        }

        df = pd.DataFrame(nutri_data)

        try:
            sample_imputed = imputer.transform(df)
            sample_scaled = scaler.transform(sample_imputed)
            prediction = model.predict(sample_scaled)
            decoded = label_encoder.inverse_transform(prediction)

            return render_template('index.html',
                prediction=decoded[0],
                product_name=product_name,
                brand_name=brand_name,
                image_url=image_url,
                claims=claims,
                ingredients_text=ingredients_text,
                nutriments=nutriments
            )

        except Exception as e:
            return render_template('index.html', error=f'Prediction failed: {str(e)}')

    else:
        return render_template('index.html', error='Failed to fetch data from OpenFoodFacts API.')

if __name__ == '__main__':
    app.run(debug=True)