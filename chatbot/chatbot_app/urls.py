from django.urls import path
from . import views

urlpatterns=[
    path('chatbot',views.chatbot,name='chatbot'),
    path('predict_chat',views.predict_chat,name='predict_chat')
]