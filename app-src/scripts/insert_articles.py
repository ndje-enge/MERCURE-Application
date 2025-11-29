import os
import django

# Configurer Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from exa_python_django_starter_kit.models import Article

# Liste des articles
articles = []

articles.append({
    "title": "Exemple de titre",
    "link": "https://example.com",
    "summary": "Résumé de l'article",
    "author": "Auteur",
    "date": "2023-10-01"
})

# Insérer les articles dans la base de données
for article_data in articles:
    Article.objects.create(
        title=article_data["title"],
        link=article_data["link"],
        summary=article_data["summary"],
        author=article_data.get("author"),
        date=article_data.get("date")
    )

print("Données insérées avec succès !")