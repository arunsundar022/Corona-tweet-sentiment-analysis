import tensorflow as tf
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
        prediction = model.predict(text)

        sentiment = prediction

        if sentiment == 0:
            return render_template('form.html',
            result = "Neutral")
        elif sentiment == 1:
            return render_template('form.html',
            result = "Positive")
        else:
            return render_template('form.html',
            result = "Negative")

#Run Application
if __name__ == '__main__':
    app.run(debug=True)