#for trans_image_model
import numpy as np
from skimage import transform

# for face_align
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

# 모델들이 사용할 수 있게 읽어온 이미지를 원하는 array로 변환
def trans_image_model(image):
    # cv2는 이미지를 읽어올때 GBR로 읽어온다. 이를 RGB로 변환
    image_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = np.array(image_for_model).astype('float32')/255

    # 안경, 선글라스, 마스크 모델용 이미지
    # RGB 3채널 + 학습사진크기 : 128 * 128
    RGBimage = transform.resize(image_for_model, (128, 128, 3))
    RGBimage = np.expand_dims(RGBimage, axis=0)

    # 감정 모델용 이미지 
    # gray 스케일로 인한 1채널 + 학습사진크기 : 48 * 48 
    GRAYimage = transform.resize(image_for_model, (48, 48, 1))
    GRAYimage = np.expand_dims(GRAYimage, axis=0)
   
    return (RGBimage, GRAYimage)


def face_align(image):
    predictor_model = "./landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x+w], width=256)
        faceAligned = fa.align(image, gray, rect)
        return faceAligned

# 이미지에서 얼굴인식하고 추출
def reg_face(image):
    
    prototxtPath = "./face_detector/deploy.prototxt"
    weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    locs = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

	    # filter out weak detections by ensuring the confidence is
	    # greater than the minimum confidence
        if confidence > 0.5:
	        # compute the (x, y)-coordinates of the bounding box for
	        # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
	    	# ensure the bounding boxes fall within the dimensions of
	    	# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            locs.append((startX, startY, endX, endY))
    
    # only make a predictions if at least one face was detected 
    if len(locs) > 1:
        return -1

    return image[locs[0][1]:locs[0][3], locs[0][0]:locs[0][2]]