from flask import Flask
from flask import request
from flask import jsonify
from joblib import load

app = Flask(__name__)
#model_path = "../models/SVC(C=10, gamma=0.0001)"
#print(model_path)
model = load('svm.joblib')

@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    print("done loading")
    predicted1 = model.predict([image1])
    image2 = request.json['image2']
    print("done loading")
    predicted2 = model.predict([image2])
    if(predicted1[0]==predicted2[0]):
        return "Same Digits"
    else :
        return "Not Same"
