# Import all libraries and modules for use during lecture session code walkthrough
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import string
import seaborn as sn
import tikzplotlib

from collections import Counter
from IPython.core.interactiveshell import InteractiveShell
from flask import Flask, abort, jsonify, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from urllib.request import urlopen
from urllib.parse import urlencode, quote_plus, urlparse
from xml.dom.minidom import parse, parseString

var_stemmer = PorterStemmer()



# 1. Case Folding
# Good idea to implement a function here [...]

def fxn_etd_case_folding(var_input):
    return var_input.lower()


# 2. Punctuation

# Function for removing stopwords from string of text
def fxn_etd_punctuation(var_input_text):
    var_output_text = re.sub("[%s]" % re.escape(string.punctuation), " ", var_input_text)
    var_output_text = re.sub("[%s]" % re.escape(string.punctuation), " ", var_output_text)
    var_output_text = re.sub('\w*\d\w*', '', var_output_text) # HINT: lookup isalpha() function
    return var_output_text
    
# 4. Stopwords

# Function for removing stopwords from string of text
def fxn_etd_stopwords(var_input_text):
    var_etd_stop = " ".join([
        var_etd_word for var_etd_word in var_input_text.split() 
        if var_etd_word not in stopwords.words('english')
    ])
    return var_etd_stop

# 5. Stemming

# Function for removing stopwords from string of text
# Remember: input will be chunck of text
def fxn_etd_stem(var_input_text):
    var_output_text = " ".join([
        var_stemmer.stem(var_etd_word) for var_etd_word in var_input_text.split() 
    ])
    return var_output_text


# --- Prediction lookups ---

# ETD collections
var_etd_collections = {0:'Education',1:'Medicine',2:'Natural Sciences',3:'Agricultural Sciences',4:'Humanities and Social Sciences',5:'Law',6:'Veterinary Medicine',7:'Engineering',8:'Mines',9:'Library',10:'Institute of Distance Education'}

# ETD types
var_etd_types = {0: 'Masters', 1: 'Doctoral'}

# ETD collection classification: Load saved tokenizer and model
var_etd_vectoriser_tfidf_data_title_abstract = joblib.load("bak-var_etd_vectoriser_tfidf_data_title_abstract.pkl")
var_classifier_sgd_title_abstract = joblib.load("bak-var_classifier_sgd_title_abstract.pkl")

# ETD type classification: Load saved tokenizer and model
var_unza_etd_vectoriser_cv_coverpages = joblib.load("bak-var_unza_etd_vectoriser_cv_coverpages.pkl")
var_classifier_rf_cover_page = joblib.load("bak-var_classifier_rf_cover_page.pkl")


app = Flask(__name__)

@app.route('/api/collection', methods=['POST'])
def ETD_collection_prediction():
    # all kinds of error checking should go here
    request_data = request.get_json(force=True)
    # Extract request fields
    var_etd_title = request_data["title"]
    var_etd_abstract = request_data["abstract"]
    # Preprocessing input
    var_etd_title_preprocessed = fxn_etd_stem(fxn_etd_stopwords(fxn_etd_punctuation(fxn_etd_case_folding(var_etd_title))))
    var_etd_abstract_preprocessed = fxn_etd_stem(fxn_etd_stopwords(fxn_etd_punctuation(fxn_etd_case_folding(var_etd_abstract))))
    # Concatenate title and abstract
    var_etd_title_abstract = var_etd_title_preprocessed + " " + var_etd_abstract_preprocessed
    var_etd_title_abstract_input = [var_etd_title_abstract]
    #
    var_etd_title_abstract_input_X = var_etd_vectoriser_tfidf_data_title_abstract.transform(var_etd_title_abstract_input)
    var_etd_title_abstract_input_Y = var_classifier_sgd_title_abstract.predict(var_etd_title_abstract_input_X)
    print (var_etd_title_abstract_input_Y, ":", type(var_etd_title_abstract_input_Y))
    print (var_etd_title_abstract_input_Y.tolist()[0], type(var_etd_title_abstract_input_Y.tolist()[0]))
    #print (var_etd_title_abstract_input_X)
    # Return predicted collection
    var_etd_collection_prediction = {"collectionCode": var_etd_title_abstract_input_Y.tolist()[0], "collectionName": var_etd_collections[var_etd_title_abstract_input_Y.tolist()[0]]}
    print(var_etd_collection_prediction)
    return jsonify(collectionPrediction=var_etd_collection_prediction)
    #####return jsonify(results={"test": "works"})

@app.route('/api/type', methods=['POST'])
def ETD_type_prediction():
    # all kinds of error checking should go here
    request_data = request.get_json(force=True)
    # Extract request fields
    var_etd_coverpage = request_data["coverpage"]
    # Preprocessing input
    var_etd_coverpage_preprocessed = fxn_etd_stopwords(fxn_etd_punctuation(fxn_etd_case_folding(var_etd_coverpage)))
    # Input value to predict
    var_etd_coverpage_preprocessed_input = [var_etd_coverpage_preprocessed]
    #
    var_etd_coverpage_preprocessed_input_X = var_unza_etd_vectoriser_cv_coverpages.transform(var_etd_coverpage_preprocessed_input)
    var_etd_coverpage_input_Y = var_classifier_rf_cover_page.predict(var_etd_coverpage_preprocessed_input_X)
    print (var_etd_coverpage_input_Y, ":", type(var_etd_coverpage_input_Y))
    print (var_etd_coverpage_input_Y.tolist()[0], type(var_etd_coverpage_input_Y.tolist()[0]))
    # Return predicted collection
    var_etd_type_prediction = {"typeCode": var_etd_coverpage_input_Y.tolist()[0], "typeName": var_etd_types[var_etd_coverpage_input_Y.tolist()[0]]}
    print(var_etd_type_prediction)
    return jsonify(typePrediction=var_etd_type_prediction)
    #####return jsonify(results={"test": "works"})

if __name__ == '__main__':
    app.run(port = 9009, debug = True)
