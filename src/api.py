from flask import Flask, Response, request
from flask_cors import CORS
import os
import pandas as pd
import pickle
app = Flask(__name__)

CORS(app)

# load training data
training_data = pd.read_csv(os.path.join('data', 'auto-mpg.csv'))

# load model
file_to_open = open(os.path.join('data', 'models','linear_regression_model.pkl'), 'rb')
trained_model = pickle.load(file_to_open)
file_to_open.close()

@app.route("/", methods=["GET"])
def index():
    return {"hello": "world"}


@app.route("/hello_world", methods=["GET"])
def hello_world():
    return "<p>Hello World!</p>"


@app.route("/training_data", methods=["GET"])
def get_training_data():
    return Response(training_data.to_json(), mimetype='application/json')


@app.route("/predict", methods=["GET"])
def predict():
    zylinder = request.args.get('zylinder')
    ps = request.args.get('ps')
    gewicht = request.args.get('gewicht')
    beschleunigung = request.args.get('beschleunigung')
    baujahr = request.args.get('baujahr')

    prediction = trained_model.predict([[zylinder, ps, gewicht, beschleunigung, baujahr]])

    return {"result": prediction[0]}
