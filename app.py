from flask import Flask , render_template , request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

#Model
model = joblib.load("lr_model.joblib")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/result" , methods=["POST"])
def result():
    fixed_acidity = float(request.form.get("fixed_acidity"))
    volatile_acidity = float(request.form.get("volatile_acidity"))
    citric_acid = float(request.form.get("citric_acid"))
    residual_sugar = float(request.form.get("residual_sugar"))
    chlorides = float(request.form.get("chlorides"))
    free_sulfur_dioxide = float(request.form.get("free_sulfur_dioxide"))
    total_sulfur_dioxide = float(request.form.get("total_sulfur_dioxide"))
    density = float(request.form.get("density"))
    pH = float(request.form.get("ph"))
    sulphates = float(request.form.get("sulphates"))
    alcohol = float(request.form.get("alcohol"))

    array = ['fixed_acidity' , 'volatile_acidity' , 'citric_acid' , 'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide' , 'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol']
    x = np.array([[fixed_acidity , volatile_acidity , citric_acid , residual_sugar , chlorides , free_sulfur_dioxide , total_sulfur_dioxide , density , pH , sulphates , alcohol]])
    
    pred = model.predict(x)[0]
    if(pred == 1): 
        pred = "Good Quality"
    else:
        pred = "Bad Quality"
    return render_template("result.html" , prediction = pred) 

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)