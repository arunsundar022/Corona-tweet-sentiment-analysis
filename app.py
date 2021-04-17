from pandas import HDFStore
from flask import Flask,request,render_template

#Load Model
model = HDFStore('LSTM Classifier.h5')

#create application
app = Flask(__name__)

#home function
@app.route('/')
def home():
    return render_template('form.html')

#Predict Function
@app.route('/',methods=['POST'])
def predict():
    text = request.form.__str__()
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
    app.run()