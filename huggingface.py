from transformers import TextClassificationPipeline
from transformers import AutoModelForSequenceClassification,AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
xlm_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment",use_fast=True)
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert-finetuned")
bert_pipe = TextClassificationPipeline(model=bert_model,tokenizer=bert_tokenizer)
xlm_model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-finetuned')
xlm_pipe = TextClassificationPipeline(model=xlm_model,tokenizer=xlm_tokenizer)

def bert_predict(text):
  if(bert_pipe([text])[0]["label"]=="LABEL_0"):
    return "Negative"
  elif(bert_pipe([text])[0]["label"]=="LABEL_1"):
    return "Neutral"
  elif(bert_pipe([text])[0]["label"]=="LABEL_2"):
    return "Positive"


def xlm_predict(text):
  if(bert_pipe([text])[0]["label"]=="LABEL_0"):
    return "Negative"
  elif(bert_pipe([text])[0]["label"]=="LABEL_1"):
    return "Neutral"
  elif(bert_pipe([text])[0]["label"]=="LABEL_2"):
    return "Positive"

