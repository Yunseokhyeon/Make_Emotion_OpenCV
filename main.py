import numpy as np
import tensorflow as tf
import cv2
from model import emotion, mask, glasses, sunglasses, skin
from util import trans_image_model, reg_face, add_face_part

# 학습된 모델 가져오기
model_emotion = emotion()
model_mask = mask()
model_glasses = glasses()
model_sunglasses = sunglasses()
model_skin = skin()

# 분류될 감정리스트
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 
                3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
skin_dict = {0: "asian", 1:"black", 2:"white"}
mask_dict = {0: "mask", 1:"noMask"}
glasses_dict = {0: "circle", 1:"noGlasses", 2:"sunglasses"}

# 대상 사진파일경로
file_path = './face/black6.png'

# 대상사진
image = cv2.imread(file_path)

image, code = reg_face(image)

if code == -1:
    print("사람이 1명 이상입니다.")
else :
    # 모델별로 쓰일 이미지 변환
    RGBimage, GRAYimage = trans_image_model(image)

    # 결과를 저장할 dictionary
    result={}
    # 초기 설정 마스크를 꼈을 경우 default Value
    # skin = asian / emotion = neutral
    result['skin'] = skin_dict[0]
    result['emotion'] = emotion_dict[4]

    # 마스크 유무 판별
    result_mask = model_mask.predict(RGBimage)
    result['mask'] = mask_dict[int(np.argmax(result_mask[0]))]

    # 마스크를 끼지 않은 경우 피부색과 감정인식을 할 수 있다.
    if result['mask'] == 'noMask' :
        # 피부색 판별
        result_skin = model_skin.predict(RGBimage)
        result['skin'] = skin_dict[int(np.argmax(result_skin[0]))]

        # 감정인식
        result_emotion = model_emotion.predict(GRAYimage)
        result['emotion'] = emotion_dict[int(np.argmax(result_emotion))]
        

    # 안경 혹은 선글라스 판별
    result_glasses = model_glasses.predict(RGBimage)
    result_sunglasses = model_sunglasses.predict(RGBimage)

    # 안경모델의 결과는 0:안경, 1:안경X
    # 선글라스모델의 결과는 0:선글라스X, 1:선글라스O
    # 따라서 선글라스인 경우 *2 를 적용해야 원하는 값을 얻는다.
    result['glasses'] = glasses_dict[int(np.argmax(result_glasses[0])) +  int(np.argmax(result_sunglasses[0])) * 2]

    print(file_path)
    print(result)
    print('#####################################')


    # 이모티콘 생성영역
    face = cv2.imread('./emoticon/face/'+ result['skin'] +'.png')
    add_face_part(face, result['emotion'], 'eye', 130, 170) # 눈붙이기

    # 마스크 착용인 경우 입 대신 마스크를 착용한다.
    if result['mask'] == 'mask':
        add_face_part(face,'mask', 'white', 50, 290)
    else :
        add_face_part(face, result['emotion'], 'mouth', 180, 350 + 90) # 입붙이기

    # 안경을 착용하지 않은 경우 안경 착용
    if result['glasses'] != 'noGlasses':
        add_face_part(face, 'glasses', result['glasses'], 110, 120) #안경

    # 결과출력
    cv2.imwrite('result' + '.png',face)

