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
    try:
        # Retrieve and convert query parameters to appropriate numeric types
        zylinder = int(request.args.get('zylinder'))
        ps = int(request.args.get('ps'))
        gewicht = float(request.args.get('gewicht'))
        beschleunigung = float(request.args.get('beschleunigung'))
        baujahr = int(request.args.get('baujahr'))

        # Make the prediction
        prediction = trained_model.predict([[zylinder, ps, gewicht, beschleunigung, baujahr]])

        # Return the result as JSON
        return {
            "input": {
                "zylinder": zylinder,
                "ps": ps,
                "gewicht": gewicht,
                "beschleunigung": beschleunigung,
                "baujahr": baujahr
            },
            "result": prediction[0]
        }

    except ValueError as e:
        # Handle conversion errors
        return {"error": str(e)}, 400

    except Exception as e:
        # Handle any other errors
        return {"error": "An error occurred: " + str(e)}, 500



    
    
