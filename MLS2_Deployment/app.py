import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder

ohe = joblib.load('encoder.joblib')

app = Flask(__name__)
model = pickle.load(open('mls2model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    Gender = request.form['Gender']
    Married = request.form['Married']
    Dependents = request.form['Dependents']
    Education = request.form['Education']
    Self_Employed = request.form['Self-Employed']
    Property_Area = request.form['Property_Area']
    LoanAmount = request.form['LoanAmount']
    Loan_Amount_Term = request.form['Loan_Amount_Term']
    Credit_History = request.form['Credit_History']
    Total_Income = request.form['Total_Income']
    
    dffeature = pd.DataFrame({'Gender': Gender,
    'Married': Married,
    'Dependents':Dependents,
    'Education': Education,
    'Self_Employed':Self_Employed,
    'Property_Area': Property_Area,
    'LoanAmount':LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History':Credit_History,
    'Total_Income':Total_Income
    }, index=[0])

    # Filter out the categorical columns into a list for easy reference later on in case you have more than a couple categorical columns
    categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area']
    # Apply ohe on newdf
    cat_ohe_new = ohe.transform(dffeature[categorical_cols])
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names(input_features = categorical_cols))
    #concat with original data and drop original columns
    df_ohe_new = pd.concat([dffeature, ohe_df_new], axis=1).drop(columns = categorical_cols, axis=1)


    prediction = model.predict(df_ohe_new)
    if prediction == 0:
        output = 'Rejected'
    else:
        output = 'Approved'

    return render_template('index.html', prediction_text='Loan Approval Prediction is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)