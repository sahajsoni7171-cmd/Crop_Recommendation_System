from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model and label encoder
model = pickle.load(open("crop_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Create Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create numpy array for prediction
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(data)
        crop = le.inverse_transform(prediction)[0]

        return render_template('index.html', result=crop)

    except Exception as e:
        return render_template('index.html', result="Error: Invalid Input")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
