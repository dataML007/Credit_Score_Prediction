from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import numpy as np
import pickle
import json
from score_code import *

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():

    json_data = request.get_json()
    df = pd.DataFrame(json_data)
    #select features and target
    X = df[df.columns.difference(['y'])]
    y = df.y
    # feature transformation
    X = X[selected_columns]
    imputed_X = X.fillna(median_values)
    X = pd.DataFrame(scaler.transform(imputed_X), columns=imputed_X.columns)

    # model evaluation
    rmse_val, acc_val, r2_val, pred = model_evaluate(reg, X, y)
    prediction = np.array(pred).tolist()
    return jsonify({'prediction': prediction, 'rmse_val': rmse_val, 'acc_val':acc_val, 'r2_val':r2_val})

if __name__ == '__main__':

    # read model objects
    with open('model_objects.pkl', 'rb') as handle:
        selected_columns, reg, scaler, median_values = pickle.load(handle)

    app.run(debug=True, host='0.0.0.0', port=5000)
