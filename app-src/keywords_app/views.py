# Create your views here.
import pandas as pd
from django.shortcuts import render, redirect
from .forms import UploadFileForm
from .models import Keyword

def upload_keywords(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Lire le fichier Excel
            file = request.FILES['file']
            xls = pd.ExcelFile(file)
            df = pd.read_excel(file, sheet_name=xls.sheet_names[0])

            # Extraire les mots-clés
            header_row = df[df.apply(lambda row: row.astype(str).str.contains('THESAURUS VEILLE REGLEMENTAIRE & ONG-PRESSE', case=False).any(), axis=1)].index[0]
            keywords = df.iloc[header_row + 1:, 0].dropna().tolist()

            # Mettre à jour la base de données
            for word in keywords:
                Keyword.objects.using('keywords_db').get_or_create(word=word)

            return redirect('upload_keywords')  # Rediriger après succès
    else:
        form = UploadFileForm()

    return render(request, 'upload_keywords.html', {'form': form})

def list_keywords(request):
    keywords = Keyword.objects.using('keywords_db').all()  # Récupérer les mots-clés depuis la base keywords_db
    return render(request, 'list_keywords.html', {'keywords': keywords})