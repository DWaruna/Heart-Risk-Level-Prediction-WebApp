from flask import Flask,render_template,request
import numpy as np
import joblib
from keras.models import load_model
from keras import backend as k

model = load_model('model/model-087.model')

scaler_data = joblib.load('model/scaler_data.sav')
scaler_target = joblib.load('model/scaler_target.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('patiant_detiles.html')

@app.route('/getresult', methods=['POST'])
def getresult():
    result = request.form
    print(result)

    name = result['name']
    gender = float(result['gender'])
    age = float(result['age'])
    tc = float(result['tc'])
    hdl = float(result['hdl'])
    smoke = float(result['smoke'])
    bpm = float(result['bpm'])
    diab = float(result['diab'])

    test_data = np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,-1)

    test_data = scaler_data.transform(test_data)

    prrediction = model.predict(test_data)

    prrediction = scaler_target.inverse_transform(prrediction)
    print(prrediction[0][0])
    resultDict = {"name":name, "risk":round(prrediction[0][0],2)}

    return render_template('patient_result.html',results=resultDict)

app.run(debug=True)