from django.urls import path
from . import views

app_name = 'image_to_text'

urlpatterns = [
    path('', views.news_classification, name='news_classification'), 

] 
