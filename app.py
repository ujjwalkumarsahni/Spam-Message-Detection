from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Get user input
        data = [message]
        seq = tokenizer.texts_to_sequences(data)  # Convert text to sequence
        padded = pad_sequences(seq, maxlen=100)  # Pad sequence to match model input
        prediction = model.predict(padded)  # Get prediction
        result = "Spam" if prediction[0][0] > 0.5 else "Not Spam"  # Convert probability to label
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
