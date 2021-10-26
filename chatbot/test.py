# query="what is chicken pox"
# query_list=query.split(' ')
# find='chicken sox'
# find_tuple=tuple(find.split(" "))
# print(find_tuple)
# lst=[]
# for i in range(len(query_list)):
#     lst.append((query_list[i]))
#     lst.append((query_list[i-2],query_list[i-1],query_list[i]))
#     lst.append((query_list[i-1],query_list[i]))
# for i in range(len(lst)):
#     if find_tuple==lst[i]:
#         print(True)
#         break
import pandas as pd
chat="What are symptoms of Peptic ulcer diseae"
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
    print("Please ask me about correct disease.")
else:
    final_disease = " ".join(temp)
    all_symptoms1=""
    df_symptoms=df_symptoms[df_symptoms['Disease']==final_disease].reset_index()
    print(df_symptoms.head(5))
    for i in range(min(df_symptoms.shape[0],5)):
        all_symptoms1+=str(i+1)+"."
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
    
    print("all_symptoms1", all_symptoms1)
    # return HttpResponse(json.dumps({'ans': precautions}), content_type="application/json")
