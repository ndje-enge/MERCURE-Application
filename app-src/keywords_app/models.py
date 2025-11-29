from django.db import models

class Keyword(models.Model):
    word = models.CharField(max_length=255)

    def __str__(self):
        return self.word

    class Meta:
        app_label = 'keywords_app'

# Create your models here.
