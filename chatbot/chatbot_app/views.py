from django.urls import reverse
from .models import *
# Create your views here.
from django.shortcuts import render, HttpResponse
import pickle
import json
import random
import numpy as np
import nltk
import pandas as pd
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import keras
# Create your views here.

disease_model= pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//Multinomial_classifier_disease.pkl','rb'))
disease_tokenizer = pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//tf_idf_vectorizer_disease.pkl','rb'))
model = keras.models.load_model('C://Users//LENOVO//projects//Health Care Chatbot//notebook//chatbot_model.h5')
intents = json.loads(open('C://Users//LENOVO//projects//Health Care Chatbot//data//intents.json').read())
words = pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//words.pkl','rb'))
classes = pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//classes.pkl','rb'))
print("---------------------You are set to go.--------------------------")
flag=False
prec,desc,sym=False,False,False
temp_disease=''
all_symptoms=''

def chatbot(request):
    return render(request,'chatbot.html')

def clean(text):
    text = text.lower() 
    text = text.split()
    text = ' '.join(text)
    return text

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result,tag

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res,tag = getResponse(ints, intents)
    return res,tag

def predict_chat(request):
    pred="please type something"
    tag=""
    global prec,desc,sym,flag,temp_disease,all_symptoms
    if request.method == 'POST':
        print('hello')
        chat=request.POST['operation']
        pred,tag=chatbot_response(chat)
        print("pred",pred,"tag",tag)
        if tag=="tell_symptoms":
            sym=True
            prec,desc=False,False
            return HttpResponse(json.dumps({'ans':"Please tell your symptoms."}), content_type="application/json")
        elif tag=="no_symptoms":
            # chat+=all_symptoms
            chat=all_symptoms
            symptoms=chat.split(',')
            symptoms=" ".join(symptoms)
            test=clean(symptoms)
            test=[test]
            test_vectorized=disease_tokenizer.transform(test)
            pred=disease_model.predict(test_vectorized)[0]
            temp_disease=pred
            sym=False
            print("test",test)
            return HttpResponse(json.dumps({'ans':"You have "+pred+"."}), content_type="application/json")
        elif tag=="add_symptoms":
            # all_symptoms+=chat+" "
            tag=''
            return HttpResponse(json.dumps({'ans':"Tell me more symptoms"}), content_type="application/json")
            
        elif sym:
            # symptoms=chat.split(',')
            # symptoms=" ".join(symptoms)
            # test=clean(symptoms)
            # test=[test]
            # test_vectorized=disease_tokenizer.transform(test)
            # pred=disease_model.predict(test_vectorized)[0]
            # temp_disease=pred
            # sym=False
            # print("test",test)
            all_symptoms+=chat+" "
            return HttpResponse(json.dumps({'ans':"have you told all symptoms to me or you want to tell me more."}), content_type="application/json")
        elif tag=="tell_precautions":
            df_precautions=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_precaution.csv")
            # temp_disease="Chicken pox"
            print("temp_disease",temp_disease)
            temp_disease=temp_disease[0].upper()+temp_disease[1:]
            print("temp_disease",temp_disease)
            
            precautions=str(df_precautions[df_precautions['Disease']==temp_disease]['Precaution_1'].values[0]+", "+df_precautions[df_precautions['Disease']==temp_disease]['Precaution_2'].values[0]+", "+df_precautions[df_precautions['Disease']==temp_disease]['Precaution_3'].values[0]+" and  "+df_precautions[df_precautions['Disease']== temp_disease]['Precaution_4'].values[0])
            print("precautions",precautions)
            return HttpResponse(json.dumps({'ans':precautions}), content_type="application/json")
        elif tag=="tell_description":
            df_description=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_Description.csv")
            # temp_disease="Chicken pox"
            print("temp_disease",temp_disease)
            temp_disease=temp_disease[0].upper()+temp_disease[1:]
            print("temp_disease",temp_disease)
            
            description=str(df_description[df_description['Disease']==temp_disease]['Description'].values[0])
            print("description",description)
            return HttpResponse(json.dumps({'ans':description}), content_type="application/json")
        # pred,tag=chatbot_response(chat)
    return HttpResponse(json.dumps({'ans':pred}), content_type="application/json")