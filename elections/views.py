from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .models import Candidate, Poll, Choice
from .makeEmoticon.emoticon import make_emoticon

import cv2
import numpy as np
import re
import base64

import datetime

#from .forms import UploadFileForm

# Create your views here.
def index(request):
		return render(request, 'elections/index.html')


def captureImage(request):
	return render(request, 'elections/captureImage.html')

def makeEmoticon(request):
	if request.method == "POST":
		captured_image = request.POST.get('captured_image')
		
		imgstr = re.search(r'base64,(.*)', captured_image).group(1)
		origin = imgstr
		imgstr = base64.b64decode(imgstr)
		encoded_img = np.fromstring(imgstr, dtype = np.uint8)
		image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

		emoticon, code = make_emoticon(image)		
		if code == -1:
			return "얼굴이 1개 이상입니다"
		elif code == -2:
			return "얼굴이 없습니다."
		ctx = {}

		image_data = base64.b64encode(cv2.imencode(".png", emoticon)[1].tostring()).decode("utf-8")
		ctx["image"] = image_data
		ctx["origin"] = origin

		return render(request, 'elections/image.html', ctx)

def imageIndex(request):
	return render(request, 'elections/upload.html')

def uploadImg(request):
	if request.method == 'POST':
		imgstr = request.FILES['uploadImg'].read()
		encoded_img = np.fromstring(imgstr, dtype = np.uint8)
		image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

		emoticon, code = make_emoticon(image)		
		if code == -1:
			return "얼굴이 1개 이상입니다"
		elif code == -2:
			return "얼굴이 없습니다."
		ctx = {}

		image_data = base64.b64encode(cv2.imencode(".png", emoticon)[1].tostring()).decode("utf-8")
		ctx["image"] = image_data

		ctx["origin"] = base64.b64encode(encoded_img).decode("utf-8")

		return render(request, 'elections/image.html', ctx)


def areas(request, area):
	today = datetime.datetime.now()
	try:
		poll = Poll.objects.get(area = area, startdate__lte=today, enddate__gte=today)
		candidates = Candidate.objects.filter(area = area)
	except:
		poll = None
		candidates = None
	context = {'candidates':candidates,
	'area':area,
	'poll': poll}
	return render(request, 'elections/area.html', context)

