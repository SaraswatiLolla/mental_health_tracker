import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect
import joblib


app = Flask(__name__)

Gbr = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    my_dict = {
        'A': 1,
        'B': 0
    }

    input = [str(x) for x in request.form.values()]
    df_1 = pd.DataFrame(input)
    print(df_1)
    x = df_1[0].map(my_dict)
    df_test = x.to_frame().T
    dep = Gbr.predict(df_test)
    if dep == 0:
        output="No depression and anxiety"
    elif dep == 1:
        output="Mild depression and anxiety"
    elif dep == 2:
        output="Moderate depression and anxiety"
    else:
        output="Severe depression and anxiety"

    return render_template('index2.html', depression_status=output)

if __name__ == "__main__":
    app.debug = True
    app.run()
