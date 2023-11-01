from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_image, name='classify_image'),
]
