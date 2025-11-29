from django.urls import path
from .views import upload_keywords, list_keywords

urlpatterns = [
    path('upload/', upload_keywords, name='upload_keywords'),
    path('list/', list_keywords, name='list_keywords'),  # URL pour afficher les mots-clés
]