import flask
from flask import Flask, render_template, request, flash
from joblib import load
import numpy as np
import os

app = Flask(__name__)

app.secret_key = os.urandom(24)

model = load('tree4.joblib')

@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == 'POST':
        passenger_class = request.form['passs']
        age = request.form['age']
        gender = request.form['gender']
        if gender == "1":
            female = 0
            male = 1
        else:
            female = 1
            male = 0
        y_pred = [[passenger_class, age, female, male]]      
        preds = model.predict(y_pred)
        if preds == 0:
            flash("No Survived", 'danger')
        else:
            flash("Survived", 'success')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

