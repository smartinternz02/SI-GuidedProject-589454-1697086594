import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
#create flask app
flask_app=Flask(__name__)
model = pickle.load(open('model.pkl','rb')) 
@flask_app.route('/')
def Home():
    return render_template('index.html')

@flask_app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    features = [np.array(int_features)]
    print(features)
    prediction = model.predict(features)
    output=prediction[0]
    return render_template("index.html",prediction_text="{} detected".format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)
    