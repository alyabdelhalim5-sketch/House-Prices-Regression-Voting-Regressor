import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load the saved model files ---
# housing_model: The trained Voting Regressor
# scaler: The StandardScaler used during training
# model_columns: The list of 278 feature names from the training set
housing_model = pickle.load(open('housing_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the main home page."""
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Handles API prediction requests.
    Accepts raw JSON data (even partial features), performs encoding/alignment,
    and returns the predicted house price.
    """
    try:
        # 1. Get data from the request body
        data = request.json['data']
        
        # 2. Convert input dictionary to DataFrame
        df = pd.DataFrame([data])
        
        # 3. Handle Categorical Encoding
        # Convert strings to dummy/indicator variables
        df_encoded = pd.get_dummies(df)
        
        # 4. Feature Alignment (Crucial Step)
        # Reindex ensures the input has the same 278 columns as the training set.
        # Missing columns are filled with 0, and extra columns are dropped.
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 5. Scaling the data
        # Transforming features to have mean=0 and variance=1
        new_data = scaler.transform(df_final)
        
        # 6. Model Prediction
        output = housing_model.predict(new_data)
        
        # Log the prediction to the console
        print(f"Success! Predicted Price: {output[0]}")
        
        return jsonify(float(output[0]))

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e), "status": "failed"})

if __name__ == '__main__':
    # Start the Flask development server
    app.run(debug=True)