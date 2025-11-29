from django.urls import path
from .views import upload_factiva,article_detail, supp2, show_factiva_by_date,append_factiva,download_classification_results, view_classified_articles,preshow_factiva_by_date, preview_classified_articles,prereporting_articles, reporting_articles, download_excel, supp, preclassify_titles_by_date


urlpatterns = [
    path('upload/', upload_factiva, name='upload_factiva'),
    path("articles/<int:article_id>/", article_detail, name="article_detail"),
    path('show/', show_factiva_by_date, name='show_factiva_by_date'),
    path('preshow/', preshow_factiva_by_date, name='preshow_factiva_by_date'),
    path('preclassify/', preclassify_titles_by_date, name='preclassify_titles_by_date'),
    path('prereporting_articles/', prereporting_articles, name='prereporting_articles'),
    path('append/', append_factiva, name='append_factiva'),
    path('download-results/', download_classification_results, name='download_classification_results'),
    path('classified-articles/', view_classified_articles, name='view_classified_articles'),
    path('preclassified-articles/', preview_classified_articles, name='preview_classified_articles'),
    path('reporting_articles/', reporting_articles, name='reporting_articles'),
    path('download-excel/', download_excel, name='download_excel'),
    path('supp/', supp, name='supp'),
    path('supp2/', supp2, name='supp2'),
]