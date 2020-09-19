from django.conf.urls import url,include
from . import views

urlpatterns = [
	url(r'^$', views.index),
	url(r'^areas/(?P<area>.+)/$', views.areas),
	url(r'^captureImage', views.captureImage),
	url(r'^makeEmoticon', views.makeEmoticon),
	url(r'^uploadImg', views.uploadImg),
	url(r'^imageIndex', views.imageIndex),
	
	
]