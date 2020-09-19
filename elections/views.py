from django.shortcuts import render
from django.http import HttpResponse

from .models import Candidate, Poll, Choice

import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import random
import os
import natsort
import warnings
warnings.filterwarnings('ignore')
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2 

from PIL import Image
from skimage import transform

predictor_model = "./landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)
fa = FaceAligner(predictor, desiredFaceWidth=256)
#image = open("./irene.png", "rb").read()

# Create your views here.
def index(request):

	# image = cv2.imread("./irene.png")
	# #gray = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY) #
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# rects = detector(gray, 2)
	# for rect in rects:
	# 	(x, y, w, h) = rect_to_bb(rect)
	# 	faceOrig = imutils.resize(image[y:y+h, x:x+w], width=256)
	# 	faceAligned = fa.align(image, gray, rect)

	# 	cv2.imwrite("alignImage.png",faceAligned)
	# 	testImage = open("./alignImage.png", "rb").read()
		return render(request, 'elections/index.html')


def captureImage(request):
	return render(request, 'elections/captureImage.html')

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

