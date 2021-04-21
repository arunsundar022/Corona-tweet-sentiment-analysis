import numpy as np
import tensorflow as tf
from nlp import nlp_ready
from tensorflow import keras
from flask import Flask,request,render_template

#Load Model
model = tf.keras.models.load_model('model.h5')

#create application
app = Flask(__name__)

#home function
@app.route('/')
def home():
    return render_template('form.html')

#Predict Function
@app.route('/',methods=['POST'])
def predict():
    if request.method == "POST":
        text = request.form.get("tex")
        text = nlp_ready(text)
        prediction = np.argmax(model.predict_classes(text))
        print(prediction)

        sentiment = prediction

        if sentiment == 0:
            return render_template('prediction.html',
            result = "Neutral")
        elif sentiment == 1:
            return render_template('prediction.html',
            result = "Positive")
        else:
            return render_template('prediction.html',
            result = "Negative")

#Run Application
if __name__ == '__main__':
    app.run(debug=True)