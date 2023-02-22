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

from transformers import TextClassificationPipeline,pipeline
from transformers import AutoModelForSequenceClassification,AutoTokenizer
# bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# xlm_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment",use_fast=True)
# bert_model = AutoModelForSequenceClassification.from_pretrained("./bert-finetuned")
# bert_pipe = TextClassificationPipeline(model=bert_model,tokenizer=bert_tokenizer)
# xlm_model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-finetuned')
# xlm_pipe = TextClassificationPipeline(model=xlm_model,tokenizer=xlm_tokenizer)
bert_pipe = pipeline("text-classification",model="cruiser/distilbert-tweet-sentiment-finetuned")
roberta_pipe = pipeline("text-classification",model="cruiser/twitter-roberta-tweet-sentiment-finetuned")

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

vect = load('new_models/vectorizer.pkl')
mnb = load('new_models/mnb.pkl')
svm = load('new_models/svm.pkl')
rfclf = load('new_models/rfclf.pkl')
lr = load("new_models/lrmodel.pkl")

def get_sentiment(x):
    if x == 1: 
        return 'Neutral'
    elif x == 0:
        return 'Negative'
    elif x==2:
        return "Positive"
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




@app.route('/test',methods=["POST"])
def sentiment():
    data = request.json
    id=data["id"]
    value = data["text"]
    def preprocess_entire_data(value,id):
        processed_input = data_preprocessing(value)
        stemmed_input = stemming(processed_input)
        vectorized = vect.transform([stemmed_input])
        prediction1 = mnb.predict(vectorized)
        prediction2 = svm.predict(vectorized)
        prediction3 = rfclf.predict(vectorized)
        prediction4 = lr.predict(vectorized)
        print("preprocessing")

        return {
            "id":id,
            "tweet":value,
            "models":[
                {
                    "model_id":0,
                    "name":"Multinomial Naive Bayes",
                    "prediction":get_sentiment(prediction1[0])
                },                {
                    "model_id":1,
                    "name":"Support Vector Machine",
                    "prediction":get_sentiment(prediction2[0])
                },{
                    "model_id":2,
                    "name":"Random Forest Classifier",
                    "prediction":get_sentiment(prediction3[0]),
                },
                {
                    "model_id:":3,
                    "name":"Logistic Regression",
                    "prediction":get_sentiment(prediction4[0])
                },
                {
                    "model_id":4,
                    "name":"BERT Finetuned",
                    "prediction":bert_pipe(value)[0]["label"]
                },
                {
                    "model_id":5,
                    "name":"Twitter Roberta Finetuned",
                    "prediction":roberta_pipe(value)[0]["label"]
                }
            ]
        }
    result = preprocess_entire_data(value,id)
    response = jsonify(result)
    # response = jsonify({"prediction1": get_sentiment(prediction1[0]), "prediction2": get_sentiment(
    #     prediction2[0]), "prediction3": get_sentiment(prediction3[0])})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



if __name__ == '__main__':
    app.run(port=5000, debug=True)
