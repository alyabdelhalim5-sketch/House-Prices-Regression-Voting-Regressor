import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load the saved model files ---
housing_model = pickle.load(open('housing_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        new_data = scaler.transform(df_final)
        output = housing_model.predict(new_data)
        
        # تحويل السعر من Log للسعر الحقيقي
        real_price = np.exp(output[0])
        
        print(f"Success! Predicted Price: {real_price}")
        return jsonify(float(real_price))

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

# تأكد إن الدالة دي بره الـ predict_api وعلى الشمال خالص
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # تصحيح request.form
        data = [float(x) for x in request.form.values()]
        # تصحيح transform
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        
        output = housing_model.predict(final_input)[0]
        real_price = np.exp(output)
        
        return render_template('home.html', prediction_text="The predicted price is ${:,.2f}".format(real_price))
    except Exception as e:
        return render_template('home.html', prediction_text="Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)