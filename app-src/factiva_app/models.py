# Create your models here.
from django.db import models
import pandas as pd

class FactivaData(models.Model):

    nom_source = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    publisher = models.CharField(max_length=255)
    code_source = models.CharField(max_length=255)
    publication_date = models.DateField()
    content = models.TextField()
    def to_dict(self):
        return {
            "title": self.title,
            "publication_date": self.publication_date,
            "source_code": self.code_source,
            "publisher": self.publisher,
            "content": self.content,
            "nom_source": self.nom_source,
        }
    def __str__(self):
        return self.title

    class Meta:
        app_label = 'factiva_app'

class ClassifiedArticle(models.Model):
    title = models.CharField(max_length=255)
    label = models.CharField(max_length=10)  # Par exemple, "oui" ou "non"
    expressions_detectees = models.TextField()
    publication_date = models.DateField()
    source_code = models.CharField(max_length=50)
    publisher = models.CharField(max_length=255, null=True, blank=True)
    content = models.TextField()
    nom_source = models.CharField(max_length=255)

    def __str__(self):
        return self.title

    def to_dict(self):
        return {
            "title": self.title,
            "label": self.label,
            "expressions_detectees": self.expressions_detectees,
            "publication_date": self.publication_date,
            "source_code": self.source_code,
            "publisher": self.publisher,
            "content": self.content,
            "nom_source": self.nom_source,
        }

    @staticmethod
    def to_dataframe(queryset):
        return pd.DataFrame([article.to_dict() for article in queryset])
