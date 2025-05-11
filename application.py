from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


application = Flask(__name__)

# Function to load pickled models
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Failed to load model {file_path}: {e}")
        return None


scaler_set  = pickle.load(open('model/scale_set_pipeline.pkl', 'rb'))
scaler_serd = pickle.load(open('model/scale_serd_pipeline.pkl', 'rb'))
scaler_som  = pickle.load(open('model/scale_som_pipeline.pkl', 'rb'))
loaded_model_set  = pickle.load(open('model/model_set_pipeline.pkl', 'rb'))
loaded_model_serd = pickle.load(open('model/model_serd_pipeline.pkl', 'rb'))
loaded_model_som  = pickle.load(open('model/model_som_pipeline.pkl', 'rb'))
default_value_set  = pickle.load(open('model/default_value_set.pkl', 'rb'))
default_value_serd = pickle.load(open('model/default_value_serd.pkl', 'rb'))
default_value_som  = pickle.load(open('model/default_value_som.pkl', 'rb'))

METRICS = {
    'SET':  {'mse': 0.0542, 'r2': 0.7824},
    'SERD': {'mse': 0.0505, 'r2': 0.6472},
    'SOM':  {'mse': 0.1346, 'r2': 0.0197}
}

@application.route('/')
def index():
    return render_template('first_page.html')

@application.route('/input_set')
def input_set():
    return render_template('input_set.html')

@application.route('/predict_cgpa_set', methods=['POST'])
# @application.route('/predict_cgpa',     methods=['POST'])
def predict_cgpa_set():
    # 1) numeric inputs
    try:    eng = float(request.form['english_score'])
    except: eng = default_value_set['english_score']
    try:    mid = float(request.form['average_midterm_grade'])
    except: mid = default_value_set['average_midterm_grade']

    # 2) maps from your dropdown to the exact dummy-column name
    region_map = {
      'South Asia':       'country_South Asia',
      'West Asia':        'country_West Asia',
      'Southeast Asia': 'country_Southeast Asia',
      'Other':            'country_Other'
    }

    donor_map = {
      'RTG':              'donor_RTG',
      'No Scholarship':   'donor_selfsupport',
      'Sponsor MOU':      'donor_sponsor with mou',
      'HMKING/HMQUEEN':   'donor_thailand (HMKING & HMQUEEN)'
      # "Other" → baseline
    }

    prev_map = {
      'Science':   'PREVIOUSDEGREE_related_Science',
      'Other':     'PREVIOUSDEGREE_related_other'
      # "Engineer" → baseline
    }

    # 3) build your feature dict with zeros everywhere
    feature_dict = {
      'english_score': eng,
      'average_midterm_grade': mid,
      'country_South Asia':      0,
      'country_Southeast Asia':  0,   # still keep this column name for consistency
      'country_West Asia':       0,
      'country_Other':           0,
      'donor_RTG':               0,
      'donor_selfsupport':       0,
      'donor_sponsor with mou':  0,
      'donor_thailand (HMKING & HMQUEEN)': 0,
      'PREVIOUSDEGREE_related_Science':     0,
      'PREVIOUSDEGREE_related_other':       0
    }

    # 4) flip on the one we found in the map (if any)
    sel_country = request.form.get('Country','')
    if sel_country in region_map:
        feature_dict[region_map[sel_country]] = 1

    sel_donor   = request.form.get('donor','')
    if sel_donor in donor_map:
        feature_dict[donor_map[sel_donor]] = 1

    sel_prev    = request.form.get('PREVIOUSDEGREE','')
    if sel_prev in prev_map:
        feature_dict[prev_map[sel_prev]] = 1

    # 5) wrap in a DataFrame and predict
    df_in = pd.DataFrame([feature_dict])
    pred = loaded_model_set.predict(df_in)[0]
    m = METRICS['SET']
    return render_template(
      'result.html',
      predicted_grade=round(pred,2),
      test_mse=round(m['mse'],2),
      test_r2 =round(m['r2'],2),
      model_name='SET'
    )




@application.route('/input_serd')
def input_serd():
    return render_template('input_serd.html')

@application.route('/predict_cgpa_serd', methods=['POST'])
# @application.route('/predict_cgpa',     methods=['POST'])
def predict_cgpa_serd():
    # 1) numeric inputs
    try:    eng  = float(request.form['english_score'])
    except: eng  = default_value_serd['english_score']
    try:    mid  = float(request.form['average_midterm_grade'])
    except: mid  = default_value_serd['average_midterm_grade']
    try:    term = int(request.form['first_term'])
    except: term = default_value_serd['first_term']

    # 2) human→dummy maps
    region_map = {
      'South Asia': 'country_South Asia',
      'West Asia':  'country_West Asia',
      'Southeast Asia': 'country_Southeast Asia',
      'Other':      'country_Other'
    }
    donor_map = {
      'RTG':            'donor_RTG',
      'No Scholarship': 'donor_selfsupport',
      'Sponsor MOU':    'donor_sponsor with mou',
      'HMKING/HMQUEEN':'donor_thailand (HMKING & HMQUEEN)'
      # 'Other' is baseline
    }

    # 3) initialize feature dict (all zeros)
    feature_dict = {
      'english_score': eng,
      'first_term': term,
      'average_midterm_grade': mid,
      'country_South Asia':      0,
      'country_Southeast Asia':  0,
      'country_West Asia':       0,
      'country_Other':           0,
      'donor_RTG':               0,
      'donor_selfsupport':       0,
      'donor_sponsor with mou':  0,
      'donor_thailand (HMKING & HMQUEEN)': 0
    }

    # 4) flip on whichever was chosen
    sel_country = request.form.get('Country','')
    if sel_country in region_map:
        feature_dict[region_map[sel_country]] = 1

    sel_donor = request.form.get('donor','')
    if sel_donor in donor_map:
        feature_dict[donor_map[sel_donor]] = 1

    # 5) DataFrame & predict
    df_in = pd.DataFrame([feature_dict])
    print(df_in)
    pred = loaded_model_serd.predict(df_in)[0]
    m = METRICS['SERD']
    return render_template(
      'result.html',
      predicted_grade=round(pred,2),
      test_mse=round(m['mse'],2),
      test_r2 =round(m['r2'],2),
      model_name='SERD'
    )



@application.route('/input_som')
def input_som():
    return render_template('input_som.html')

@application.route('/predict_cgpa_som', methods=['POST'])
# @application.route('/predict_cgpa',     methods=['POST'])
def predict_cgpa_som():
    print("SOM NA JA")
    # 1) numeric inputs
    try:    age    = float(request.form['age'])
    except: age    = default_value_som['age']
    try:    eng    = float(request.form['english_score'])
    except: eng    = default_value_som['english_score']
    try:    mid    = float(request.form['average_midterm_grade'])
    except: mid    = default_value_som['average_midterm_grade']
    try:    gender = int(request.form['gender'])
    except: gender = default_value_som['gender']

    # 2) human→dummy maps
    region_map = {
      'South Asia': 'country_South Asia',
      'West Asia':  'country_West Asia', 
      'Southeast Asia': 'country_Southeast Asia',
      'Other':      'country_Other'
      # 'West Asia' → baseline
    }
    donor_map = {
      'RTG':            'donor_RTG',
      'No Scholarship': 'donor_selfsupport',
      'Sponsor MOU':    'donor_sponsor with mou',
      'HMKING/HMQUEEN':'donor_thailand (HMKING & HMQUEEN)'
    }
    prev_map = {
      'Science': 'PREVIOUSDEGREE_related_Science',
      'Other':   'PREVIOUSDEGREE_related_other'
    }

    # 3) init all-zero dict
    feature_dict = {
      'age': age,
      'gender': gender,
      'english_score': eng,
      'average_midterm_grade': mid,
      'country_South Asia':      0,
      'country_Southeast Asia':  0,
      'country_West Asia':       0,
      'country_Other':           0,
      'donor_RTG':               0,
      'donor_selfsupport':       0,
      'donor_sponsor with mou':  0,
      'donor_thailand (HMKING & HMQUEEN)': 0,
      'PREVIOUSDEGREE_related_Science': 0,
      'PREVIOUSDEGREE_related_other':   0
    }

    # 4) flip on selected
    sel_country = request.form.get('Country','')
    if sel_country in region_map:
        feature_dict[region_map[sel_country]] = 1

    sel_donor  = request.form.get('donor','')
    if sel_donor in donor_map:
        feature_dict[donor_map[sel_donor]] = 1

    sel_prev   = request.form.get('PREVIOUSDEGREE','')
    if sel_prev in prev_map:
        feature_dict[prev_map[sel_prev]] = 1

    # 5) DataFrame & predict
    df_in = pd.DataFrame([feature_dict])
    print(df_in.to_string())
    pred = loaded_model_som.predict(df_in)[0]
    m = METRICS['SOM']
    return render_template(
      'result.html',
      predicted_grade=round(pred,2),
      test_mse=round(m['mse'],2),
      test_r2 =round(m['r2'],2),
      model_name='SOM'
    )



if __name__ == '__main__':
    application.run(debug=True)
