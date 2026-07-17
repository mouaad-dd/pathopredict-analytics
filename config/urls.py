"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from prediction_app import views as prediction_views  # On importe les vues de ton app

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # 1. Page d'introduction (Accessible à la racine si l'utilisateur n'est pas connecté)
    path('', prediction_views.intro_view, name='intro'),
    
    # 2. Console de prédiction principale (l'application)
    path('analytics/', prediction_views.index, name='index'),
    
    # 3. Authentification (Connexion, Inscription et Déconnexion)
    path('login/', prediction_views.CustomLoginView.as_view(template_name='prediction_app/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', prediction_views.signup_view, name='signup'),
]