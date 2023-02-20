import requests
import numpy as np
from flask import Flask, request, jsonify
from joblib import load
import nltk
from multiprocessing import Pool
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords
from flask_cors import cross_origin, CORS
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
import json

def data_preprocessing(text):
    text = text.lower()
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


stemmer = PorterStemmer()


def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


app = Flask(__name__)
cors = CORS(app)

vect = load('vectorizer.pkl')
mnb = load('mnb.pkl')
svm = load('svm.pkl')
rfclf = load('rfclf.pkl')


def get_sentiment(x):
    if x == 1:
        return 'Positive'
    elif x == 0:
        return 'Negative'
    else:
        return 'Invalid'



    

@app.route('/api', methods=['POST'])
def predict():
    args = request.args
    search = args.get("search",type=str,default="")
    # data = request.get_json(force=True)
    # print(data)
    # value = data['input']
    # print(data['input'])
    print(search)
    url = "https://api.twitter.com/2/tweets/search/recent?max_results=10&query=" + search + " -is:retweet -has:links -has:mentions lang:en"
    print(url)
    payload={}
    headers = {
    'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANULlgEAAAAAXOTsDrOEK4PI%2BbH896TomP%2FiGLE%3DPwpbppKix6JP0xNmzqjeo580M36evkckKmPgSIwvjokZs3T4ff',
    'Cookie': 'guest_id=v1%3A167656688150839753'
    }   
    final = []
    def preprocess_entire_data(value,id):
        processed_input = data_preprocessing(value)
        stemmed_input = stemming(processed_input)
        vectorized = vect.transform([stemmed_input])
        prediction1 = mnb.predict(vectorized)
        prediction2 = svm.predict(vectorized)
        prediction3 = rfclf.predict(vectorized)
        print("preprocessing")
        final.append({"prediction1": get_sentiment(prediction1[0]), "prediction2": get_sentiment(
            prediction2[0]), "prediction3": get_sentiment(prediction3[0]),"text":value,"id":id})



    urlresponse = requests.request("GET", url, headers=headers, data=payload)
    print("we are here")
    res=json.loads(urlresponse.text)
    # processed_input = data_preprocessing(value)
    # stemmed_input = stemming(processed_input)
    # vectorized = vect.transform([stemmed_input])
    # prediction1 = mnb.predict(vectorized)
    # prediction2 = svm.predict(vectorized)
    # prediction3 = rfclf.predict(vectorized)

    # prediction1 = mnb.predict([[np.array(data['input'])]])
    # prediction2 = mnb.predict([[np.array(data['input'])]])
    # prediction3 = mnb.predict([[np.array(data['input'])]])
    print(res)
    res = res["data"]
    pool = Pool(processes=3)
    for data in res:
        pool.apply_async(preprocess_entire_data,args=(data["text"],data["id"]))
        print("pools")
    pool.close()
    pool.join()
    print("what am I doing")
    response = jsonify(final)
    # response = jsonify({"prediction1": get_sentiment(prediction1[0]), "prediction2": get_sentiment(
    #     prediction2[0]), "prediction3": get_sentiment(prediction3[0])})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(port=5000, debug=True)
