from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
app_name = 'image_classification'
urlpatterns = [
    path('', views.index, name='index'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)