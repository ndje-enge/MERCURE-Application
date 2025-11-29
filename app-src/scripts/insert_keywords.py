import os
import django
import pandas as pd

# Configurer Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from keywords_app.models import Keyword

file_path = r"C:\Travail\Data\xlsx\Mots clés3.xlsx"
xls = pd.ExcelFile(file_path)

# Examiner les noms des feuilles disponibles
sheet_names = xls.sheet_names

# Charger la première feuille
df = pd.read_excel(file_path, sheet_name=sheet_names[0])


# Localiser la première ligne contenant "THESAURUS VEILLE REGLEMENTAIRE & ONG-PRESSE"
header_row = df[df.apply(lambda row: row.astype(str).str.contains('THESAURUS VEILLE REGLEMENTAIRE & ONG-PRESSE', case=False).any(), axis=1)].index[0]

# Extraire les mots de la première colonne en dessous du titre
keywords = df.iloc[header_row + 1:, 0].dropna().tolist()

for word in keywords:  # Remplacez 'keywords' par le nom de la colonne
    Keyword.objects.using('keywords_db').create(word=word)

print("Mots-clés insérés avec succès !")
