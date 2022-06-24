

from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pickle
pipe=pickle.load(open('Wine.pkl', 'rb'))
app=Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    feature_extraction=[float(x) for x in request.form.values()]
    features=np.array([feature_extraction],dtype='object').reshape(1,11)
    prediction=pipe.predict(features)
    return render_template('index.html', prediction_text="The predicted quality of wine is{}").format(prediction)


if __name__=='__main__':
    app.run(debug=True)
