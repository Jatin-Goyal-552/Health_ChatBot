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
import random
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


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
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
        temp=""
        for c in chat:
            if c!="?" and c!="." and c!='!':
                temp+=c
        print("temp",temp)
        chat=temp
        if tag=="tell_symptoms":
            sym=True
            prec,desc=False,False
            ans_list=['Can you tell me about your symptoms?','Please tell me your symptoms, so that I can help you.']
            return HttpResponse(json.dumps({'ans':random.choice(ans_list)}), content_type="application/json")
        elif tag=="what":
            df_description=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_Description.csv")
            chat=chat.split(' ')
            all_disease=df_description['Disease'].values.tolist()
            lst=[]
            find_disease=False
            for i in range(len(chat)):
                lst.append([chat[i]])
                lst.append([chat[i-2],chat[i-1],chat[i]])
                lst.append([chat[i-1],chat[i]])
            for disease in all_disease:
                disease_list=disease.split(' ')
                for l in lst:
                    if l==disease_list:
                        temp=disease_list
                        find_disease=True
                        break
            if not find_disease:
                return HttpResponse(json.dumps({'ans':"Sorry, I do not know about this disease."}), content_type="application/json")
            final_disease=" ".join(temp)
            description=str(df_description[df_description['Disease']==final_disease]['Description'].values[0])
            print("description",description)
            return HttpResponse(json.dumps({'ans':description}), content_type="application/json")
        elif tag=="cure":
            df_precautions=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_precaution.csv")
            df_precautions=df_precautions.fillna('')
            all_disease=df_precautions['Disease'].values.tolist()
            chat=chat.split(" ")
            lst=[]
            find_disease=False
            for i in range(len(chat)):
                lst.append([chat[i]])
                lst.append([chat[i-2],chat[i-1],chat[i]])
                lst.append([chat[i-1],chat[i]])
            for disease in all_disease:
                disease_list=disease.split(' ')
                for l in lst:
                    if l==disease_list:
                        temp=disease_list
                        find_disease=True
                        break
            if not find_disease:
                return HttpResponse(json.dumps({'ans':"Sorry, I do not know about this disease."}), content_type="application/json")
            final_disease=" ".join(temp)
            precautions=str(df_precautions[df_precautions['Disease']==final_disease]['Precaution_1'].values[0]+", "+df_precautions[df_precautions['Disease']==final_disease]['Precaution_2'].values[0]+", "+df_precautions[df_precautions['Disease']==final_disease]['Precaution_3'].values[0]+" and  "+df_precautions[df_precautions['Disease']== final_disease]['Precaution_4'].values[0])
            precautions="You should take precaution like "+precautions
            print("precautions",precautions)
            return HttpResponse(json.dumps({'ans':precautions}), content_type="application/json")
        elif tag=="symptoms":
            chat=chat.split(' ')
            df_precautions = pd.read_csv(
                "C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_precaution.csv")
            df_symptoms=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//dataset.csv")
            df_symptoms=df_symptoms.fillna('')
            all_disease = df_precautions['Disease'].values.tolist()
            lst = []
            find_disease = False
            for i in range(len(chat)):
                lst.append([chat[i]])
                lst.append([chat[i-2], chat[i-1], chat[i]])
                lst.append([chat[i-1], chat[i]])
            for disease in all_disease:
                disease_list = disease.split(' ')
                for l in lst:
                    if l == disease_list:
                        temp = disease_list
                        find_disease = True
                        break
            if not find_disease:
                return HttpResponse(json.dumps({'ans':"Sorry, I do not know about this disease."}), content_type="application/json")
            else:
                final_disease = " ".join(temp)
                all_symptoms1=""
                df_symptoms=df_symptoms[df_symptoms['Disease']==final_disease].reset_index()
                print(df_symptoms.head(5))
                for i in range(1):
                    flag=True
                    for j in range(1,18):
                        col='Symptom_'+str(j)
                        if df_symptoms[col][i]!='':
                            if flag:
                                all_symptoms1+=df_symptoms[col][i]
                                flag=False
                            else:
                                all_symptoms1+=", "+df_symptoms[col][i]
                                
                    all_symptoms1+='\n'
                all_symptoms1=all_symptoms1.split(",")
                all_symptoms2=''
                for symptom in all_symptoms1:
                    symptom=symptom.split('_')
                    symptom.append(", ")
                    all_symptoms2+=" ".join(symptom)
                all_symptoms2="Symptoms for this disease is "+ all_symptoms2
                print("all_symptoms1", all_symptoms2)
                return HttpResponse(json.dumps({'ans': all_symptoms2}), content_type="application/json")

        elif tag=="no_symptoms":
            # chat+=all_symptoms
            
            temp=all_symptoms.split(",")
            all_symptoms=''
            for symptom in temp:
                symptom=symptom.strip()
                symptom=symptom.split(' ')
                all_symptoms+="_".join(symptom)
                all_symptoms+=' '
            print("all_symptoms",all_symptoms)
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
            all_symptoms=''
            ans_list=['You have '+pred+".","You are suffering from "+pred+"."]
            return HttpResponse(json.dumps({'ans':random.choice(ans_list)}), content_type="application/json")
        elif tag=="add_symptoms":
            # all_symptoms+=chat+" "
            tag=''
            ans_list=["Please tell me more symptoms.","Please add more symptoms."]
            return HttpResponse(json.dumps({'ans':random.choice(ans_list)}), content_type="application/json")
            
        elif sym:
            all_symptoms+=chat+","
            ans_list=['Have you told all symptoms to me or you want to tell me more symptom?','Do you want to tell more symptom to me?']
            return HttpResponse(json.dumps({'ans':random.choice(ans_list)}), content_type="application/json")
        elif tag=="tell_precautions":
            df_precautions=pd.read_csv("C://Users//LENOVO//projects//Health Care Chatbot//data//DiseaseData//symptom_precaution.csv")
            # temp_disease="Chicken pox"
            df_precautions=df_precautions.fillna('')
            print("temp_disease",temp_disease)
            temp_disease=temp_disease[0].upper()+temp_disease[1:]
            print("temp_disease",temp_disease)
            
            precautions=str(df_precautions[df_precautions['Disease']==temp_disease]['Precaution_1'].values[0]+", "+df_precautions[df_precautions['Disease']==temp_disease]['Precaution_2'].values[0]+", "+df_precautions[df_precautions['Disease']==temp_disease]['Precaution_3'].values[0]+" and  "+df_precautions[df_precautions['Disease']== temp_disease]['Precaution_4'].values[0])
            precautions="You should take precaution like "+precautions
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