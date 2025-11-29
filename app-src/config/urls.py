"""
URL Configuration
"""
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

urlpatterns = [
    path('', include('exa_python_django_starter_kit.urls')),
    # SWAGGER
    path('swagger/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('swagger/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('swagger/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    path('factiva/', include('factiva_app.urls')),
]
