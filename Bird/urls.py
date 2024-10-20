"""
URL configuration for Bird project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
# from django.contrib import admin
from django.urls import path
from BirdApp.views import error_view, index_view, ner_view, detail_view, overview_view, relation_view, decisions_making

urlpatterns = [
    #    path('admin/', admin.site.urls),
    path('', index_view.index, name='index'),
    path('ER-post/', ner_view.ner_llm, name='ER-post'),
    path('detail/', detail_view.showdetail, name='detail'),
    path('overview/', overview_view.show_overview, name='overview'),
    path('404/', error_view._404_, name='_404_'),
    path('search_entity/', relation_view.search_entity, name='search_entity'),
    path('search_relation/', relation_view.search_relation, name='search_relation'),
    path('decision/', decisions_making.decisions_making, name='decision')
]
