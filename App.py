import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model_log = pickle.load(open("model_log.pkl", "rb"))
model_XGB = pickle.load(open("model_XGB.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction_log = model_log.predict(features)
    prediction_xgb = model_XGB.predict(features)

    if prediction_log == 1:
        return render_template("index.html", text_log="The flight is delayed", text_xgb=prediction_xgb)
    else:
        return render_template("index.html", text_log="The flight is not delayed")


if __name__ == "__main__":
    flask_app.run(debug=True)

