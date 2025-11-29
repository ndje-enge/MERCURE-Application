import pandas as pd
from django.shortcuts import render, redirect
from factiva_app.models import ClassifiedArticle
from factiva_app.models import FactivaData
from keywords_app.models import Keyword
from .forms import UploadFileForm
from datetime import datetime
import torch
import torch.nn.functional as F
import ast 
import json
from django.http import JsonResponse
import os
import numpy as np 
import re
import time
from httpx import Client
from openai import OpenAI
import os
import pandas as pd
import time
import glob
import concurrent.futures
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from factiva_app.langgraph_classifier import process_article_with_langgraph
from httpx import Client
from django.http import HttpResponse
from io import BytesIO



def upload_factiva(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Lire le fichier Excel
            file = request.FILES['file']
            df = pd.read_excel(file)

            # Vérifier que le fichier contient les colonnes nécessaires
            required_columns = ['title', 'source_name', 'publisher_name', 'source_code', 'publication_date', 'content']
            if not all(col in df.columns for col in required_columns):
                return render(request, "upload_factiva.html", {"form": form, "error": "Le fichier doit contenir les colonnes nécessaires."})

            # Supprimer toutes les données existantes
            FactivaData.objects.using('factiva_db').all().delete()

            # Ajouter les nouvelles données
            for _, row in df.iterrows():
                FactivaData.objects.using('factiva_db').create(
                    title=row['title'],
                    nom_source=row['source_name'],
                    publisher=row['publisher_name'],
                    code_source=row['source_code'],
                    publication_date=row['publication_date'],
                    content=row['content']
                )

            return redirect('upload_factiva')  # Rediriger après succès
    else:
        form = UploadFileForm()

    return render(request, "upload_factiva.html", {"form": form})

def append_factiva(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Lire le fichier Excel
            file = request.FILES['file']
            df = pd.read_excel(file)

            # Vérifier que le fichier contient les colonnes nécessaires
            required_columns = ['title', 'source_name', 'publisher_name', 'source_code', 'publication_date', 'content']
            if not all(col in df.columns for col in required_columns):
                return render(request, "append_factiva.html", {"form": form, "error": "Le fichier doit contenir les colonnes nécessaires."})

            # Ajouter les nouvelles données tout en évitant les doublons
            for _, row in df.iterrows():
                FactivaData.objects.using('factiva_db').get_or_create(
                    title=row['title'],
                    nom_source=row['source_name'],
                    publisher=row['publisher_name'],
                    code_source=row['source_code'],
                    publication_date=row['publication_date'],
                    content=row['content']
                )

            return redirect('append_factiva')  # Rediriger après succès
    else:
        form = UploadFileForm()

    return render(request, "append_factiva.html", {"form": form})

def factiva_results(request):   
    factiva_data = FactivaData.objects.using('factiva_db').all()
    return render(request, "factiva_results.html", {"factiva_data": factiva_data})


def article_detail(request, article_id):
    article = FactivaData.objects.using('factiva_db').get(id=article_id)
    return render(request, "article_detail.html", {"article": article})


def show_factiva_by_date(request):
    factiva_data = FactivaData.objects.all()
    if request.method == "POST" and len(factiva_data) > 0:
        start_date = request.POST.get("start_date")
        end_date = request.POST.get("end_date")
        selected_sources = request.POST.getlist("sources")

        # Filtrer par période de dates
        if start_date and end_date:
            factiva_data = factiva_data.filter(publication_date__range=[start_date, end_date])

        # Filtrer par sources si "Tous" n'est pas sélectionné
        if "all" not in selected_sources:
            factiva_data = factiva_data.filter(nom_source__in=selected_sources)

        # Filtrer par pertinence
       
        
        # Ajouter une colonne "id" avec des valeurs uniques
        factiva_data_dicts = [article.to_dict() for article in factiva_data]
        # for idx, article in enumerate(factiva_data_dicts, start=1):
        #     article["id"] = idx  # Ajouter une colonne "id" avec des valeurs uniques

        # Générer un fichier Excel
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            df = pd.DataFrame(factiva_data_dicts)  # Utiliser directement la liste de dictionnaires avec "id"
            df.to_excel(writer, index=False, sheet_name="Classified Articles")
        excel_file.seek(0)  # Revenir au début du fichier pour la lecture

        # Convertir le DataFrame en une liste de dictionnaires
        factiva_data = df.to_dict(orient="records")
        nombre_articles = len(factiva_data)

        sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
        request.session['excel_file'] = excel_file.getvalue().decode('latin1')  # Stocker en tant que chaîne pour éviter les problèmes de binaire

        return render(request, "factiva_results.html", {
            "start_date" : start_date,
            "end_date" : end_date,
            "factiva_data": factiva_data,
            "sources": sources,
            "nombre_articles" : nombre_articles,
        })

    # Récupérer toutes les sources disponibles pour le formulaire
    sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
    return render(request, "show_factiva.html", {
        "sources": sources,
    })


def supp(request):
    # Supprimer toutes les données de la base de données FactivaData
    FactivaData.objects.using('factiva_db').all().delete()
    return HttpResponse("Toutes les données ont été supprimées avec succès.")

def supp2(request):
    
    ClassifiedArticle.objects.using('factiva_db').all().delete()
    return HttpResponse("Toutes les données ont été supprimées avec succès.")

def preshow_factiva_by_date(request):
    # Récupérer toutes les sources disponibles dans la colonne 'code_source'
    sources = FactivaData.objects.using('factiva_db').values_list('nom_source', flat=True).distinct()
    return render(request, "show_factiva.html" , {"sources": sources})
def preclassify_titles_by_date(request):
    # Récupérer toutes les sources disponibles dans la colonne 'code_source'
    sources = FactivaData.objects.using('factiva_db').values_list('nom_source', flat=True).distinct()
    return render(request, "classify_factiva.html", {"sources": sources})
def preview_classified_articles(request):
    # Récupérer toutes les sources disponibles dans la colonne 'code_source'
    sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
    return render(request, "classified_articles.html", {"sources": sources})
def prereporting_articles(request):
    # Récupérer toutes les sources disponibles dans la colonne 'code_source'
    sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
    return render(request, "reporting_articles.html", {"sources": sources})



def download_classification_results(request):
    # Récupérer les données classifiées depuis la session ou la base de données
    classified_data = request.session.get("classified_data", [])

    # Vérifier si des données sont disponibles
    if not classified_data:
        return HttpResponse("Aucun résultat de classification disponible pour le téléchargement.", status=400)

    # Créer un DataFrame pandas à partir des données classifiées
    df = pd.DataFrame(classified_data)

    # Ajouter une colonne "Contenu" si elle n'est pas déjà incluse
    if "content" not in df.columns:
        df["content"] = [data.get("content", "Non disponible") for data in classified_data]

    # Créer un fichier Excel en mémoire
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = 'attachment; filename="classification_results.xlsx"'

    with pd.ExcelWriter(response, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Résultats")

    return response


# ------------------- Configuration -------------------
# PROXY_source = "http://127.0.0.1:3128"
API_KEY = os.environ["OPENAI_API_KEY"]

# http_client = Client(proxy=PROXY_source, verify=False)

llm = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-5-nano",
    temperature=0,
    # http_client=http_client
)



embedding = HuggingFaceEmbeddings(
    model_name='/Users/engenouadje/Desktop/Bureau/Projets perso/roberta',
    model_kwargs={
        "device": "cpu",
        "trust_remote_code": True
    }
)


# ------------------- Fonctions Utiles -------------------
def chunk_document(doc, chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(doc)

def rag_retriever(documents, embedding):
    vector_store = FAISS.from_documents(documents, embedding)
    retriever = vector_store.as_retriever()
    return retriever

def get_rag_prompt():
    classifier_prompt = """
    Tu es un service de réponse répondant par oui ou par non. Compte tenu du contexte fourni,
    réponds à la question ci-dessous. Prends ton temps et trouve la bonne réponse dans le contexte. Réponds uniquement par « Oui » ou « Non ».

    <context>
    {context}
    </context>

    Question: {question}
    """
    return ChatPromptTemplate.from_template(classifier_prompt)

def generate_questions(labels: List[str]) -> List[tuple]:
    question_template = "Est ce que cet article parle de '{}' de manière significative?"
    return [(label, question_template.format(label)) for label in labels]

def filter_preds(preds):
    return [p[0] for p in preds if p[1].strip().lower() == 'oui']

def process_article(row: pd.Series, labels: List[str]) -> Dict:
    """
    Process article using LangGraph workflow from langgraph_classifier.py.

    """
    return process_article_with_langgraph(
        row=row,
        keywords=labels,
        llm=llm,  # ChatOpenAI instance defined at module level
        embedding=embedding  # HuggingFaceEmbeddings instance defined at module level
    )
def predict_all_articles_parallel(df: pd.DataFrame, labels: List[str], max_workers=1, progress_callback=None) -> pd.DataFrame:
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_article, row, labels) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            time.sleep(0.5)
            results.append(future.result())
           
    return pd.DataFrame(results)


def reporting_articles(request):

    if request.method == "POST":
        
        start_date = request.POST.get("start_date")
        end_date = request.POST.get("end_date")
        # Récupérer les sources sélectionnées
        selected_sources = request.POST.getlist("sources")
        # Récupérer la pertinence sélectionnée
        selected_relevance = request.POST.get("relevance")
        
        progress = 0
        def update_progress(value):
            nonlocal progress
            progress = value  
            return JsonResponse({"progress": progress})
        
        # Filtrer par période de dates
        if start_date and end_date:
             factiva_data = FactivaData.objects.using('factiva_db').filter(publication_date__range=[start_date, end_date])
        else:
             factiva_data = FactivaData.objects.using('factiva_db').all()

        # Filtrer par sources si "Tous" n'est pas sélectionné
        if "all" not in selected_sources:
            factiva_data = factiva_data.filter(nom_source__in=selected_sources)
        
        factiva_list = []
        for item in factiva_data :
            factiva_list.append({
                "id": item.id,
                "title": item.title,
                "publication_date": item.publication_date.strftime("%Y-%m-%d"),  # Convertir en chaîne
                "source_code": item.code_source,
                "publisher": item.publisher,
                "content": item.content,
                "nom_source": item.nom_source,
                "source": item.nom_source,  # Ajout du champ "source" attendu par process_article_with_langgraph
                })

        keywords_db = Keyword.objects.using("keywords_db").all()
        keywords = []
        for item in keywords_db:
            keywords.append(item.word)


        df_factiva_data = pd.DataFrame(factiva_list)

        start_time = time.time()
        df_results = predict_all_articles_parallel(df_factiva_data, keywords, max_workers=3, 
        progress_callback=update_progress
        ) 
    
        total_tokens_used = df_results["tokens_used"].sum()
        total_cost_used = df_results["total_cost"].sum()
        
        # Convertir expressions_detectees de liste en chaîne formatée pour l'affichage et la sauvegarde
        def format_expressions(expr_list):
            """Convertit une liste d'expressions en chaîne formatée."""
            if isinstance(expr_list, list):
                return ", ".join(expr_list) if expr_list else ""
            return str(expr_list) if expr_list else ""
        
        # Convertir expressions_detectees avant de générer l'Excel et de trier
        df_results["expressions_detectees"] = df_results["expressions_detectees"].apply(format_expressions)
        
        # Supprimer les colonnes de métadonnées et champs supplémentaires pour l'Excel
        # (confidence_scores et reasoning sont des métadonnées de LangGraph non nécessaires pour l'Excel)
        columns_to_drop = ["total_cost", "tokens_used", "confidence_scores", "reasoning"]
        # Ne supprimer que les colonnes qui existent
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_results.columns]
        df_results_excel = df_results.drop(columns=existing_columns_to_drop)
        
        # Ajouter une colonne "label_priority" pour trier les articles avec "oui" en premier
        df_results_excel["label_priority"] = df_results_excel["label"].apply(lambda x: 1 if x == "oui" else 0)

        # Ajouter une colonne "expressions_count" pour compter le nombre d'éléments (basé sur la chaîne)
        df_results_excel["expressions_count"] = df_results_excel["expressions_detectees"].apply(
            lambda x: len(x.split(", ")) if x and x.strip() else 0
        )

        # Trier par "label_priority" (descendant) et "expressions_count" (descendant)
        df_results_excel = df_results_excel.sort_values(by=["label_priority", "expressions_count"], ascending=[False, False])

        # Supprimer les colonnes temporaires après le tri
        df_results_excel = df_results_excel.drop(columns=["label_priority", "expressions_count"])

        # Générer un fichier Excel
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            df_results_excel.to_excel(writer, index=False, sheet_name="Classified Articles")
        excel_file.seek(0)  # Revenir au début du fichier pour la lecture
        
        # Utiliser df_results_excel pour les données classifiées (sans les colonnes de métadonnées)
        df_results = df_results_excel

        # Convertir le DataFrame en une liste de dictionnaires
        classified_data = df_results.to_dict(orient="records")
        
        nombre_articles = len(classified_data)

        sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
        
        request.session['excel_file'] = excel_file.getvalue().decode('latin1')  # Stocker en tant que chaîne pour éviter les problèmes de binaire
        
        for item in classified_data:
            
            # Convertir publication_date de chaîne en date si nécessaire
            publication_date = item["publication_date"]
            if isinstance(publication_date, str):
                publication_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
            
            # Vérifier si l'article existe déjà dans la base de données
            if not ClassifiedArticle.objects.filter(title=item["title"], publication_date=publication_date).exists():
                # Ajouter l'article à la base de données
                # expressions_detectees est maintenant une chaîne formatée
                # Note: Le champ "source" retourné par LangGraph n'est pas sauvegardé car il n'existe pas dans le modèle
                ClassifiedArticle.objects.create(
                    title=item["title"],
                    label=item["label"],
                    expressions_detectees=item["expressions_detectees"],
                    publication_date=publication_date,
                    source_code=item["source_code"],
                    publisher=item["publisher"],
                    content=item["content"],
                    nom_source=item["nom_source"],
                )

        if selected_relevance == "oui":
            classified_data = [data for data in classified_data if data["label"] == "oui"]
        elif selected_relevance == "non":
            classified_data = [data for data in classified_data if data["label"] == "non"]

        if not classified_data:
            return HttpResponse("Aucun article trouvé dans la Base de données Factiva", status=400)
        
        elapsed_time = time.time() - start_time

        if elapsed_time < 60:
            temps = f"{elapsed_time:.2f} secondes"
        elif elapsed_time < 3600:
            temps = f"{elapsed_time / 60:.2f} minutes"
        else:
            temps = f"{elapsed_time / 3600:.2f} heures"
        return render(request, "reporting_articles_results.html", {
                "classified_data": classified_data,
                "start_date": start_date,
                "end_date": end_date,
                "selected_sources": selected_sources,
                "temps" : temps,
                "nombre_articles" : nombre_articles,
                "total_tokens_used" : total_tokens_used,
                "total_cost_used" : f"{total_cost_used:.2f}",
            })
    
    sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()

    return render(request, "reporting_articles.html", {
        "sources": sources,
    })


def download_excel(request):
    excel_file = request.session.get('excel_file')
    if not excel_file:
        return HttpResponse("Aucun fichier Excel disponible pour le téléchargement.", status=400)

    response = HttpResponse(
        excel_file.encode('latin1'),  # Convertir la chaîne en binaire
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response['Content-Disposition'] = 'attachment; filename="classified_articles.xlsx"'
    return response



def view_classified_articles(request):
    classified_articles = ClassifiedArticle.objects.using('factiva_db').all()
    if request.method == "POST" and len(classified_articles) > 0:
        start_date = request.POST.get("start_date")
        end_date = request.POST.get("end_date")
        selected_sources = request.POST.getlist("sources")
        selected_relevance = request.POST.get("relevance")  # Récupérer la pertinence sélectionnée

        # Filtrer par période de dates
        if start_date and end_date:
            classified_articles = classified_articles.filter(publication_date__range=[start_date, end_date])

        # Filtrer par sources si "Tous" n'est pas sélectionné
        if "all" not in selected_sources:
            classified_articles = classified_articles.filter(source_code__in=selected_sources)

        # Filtrer par pertinence
        if selected_relevance == "oui":
            classified_articles = classified_articles.filter(label="oui")
        elif selected_relevance == "non":
            classified_articles = classified_articles.filter(label="non")
        
        
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            df = pd.DataFrame([article.to_dict() for article in classified_articles]) # Utiliser directement la liste de dictionnaires
            df["label_priority"] = df["label"].apply(lambda x: 1 if x == "oui" else 0)  # Ajouter une colonne de priorité
            df = df.sort_values(by=["label_priority"], ascending=[False]).drop(columns=["label_priority"])
            df.to_excel(writer, index=False, sheet_name="Classified Articles")
        excel_file.seek(0)  # Revenir au début du fichier pour la lecture
        
        classified_articles = df.to_dict(orient="records")  # Convertir le DataFrame en une liste de dictionnaires
        nombre_articles = len(classified_articles)
        sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
        request.session['excel_file'] = excel_file.getvalue().decode('latin1')  # Stocker en tant que chaîne pour éviter les problèmes de binaire
        return render(request, "view_classified_articles.html", {
            "classified_articles": classified_articles,
            "sources": sources,
            "start_date": start_date,
            "end_date": end_date,
            "selected_sources" : selected_sources,
            "nombre_articles" : nombre_articles,
        })

    # Récupérer toutes les sources disponibles pour le formulaire
    sources = FactivaData.objects.values_list('nom_source', flat=True).distinct()
    return render(request, "classified_articles.html", {
        "sources": sources,
    })