import numpy as np
import tensorflow as tf
import cv2
from model import emotion, mask, glasses, sunglasses
from util import trans_image_model, face_align, reg_face

# 학습된 모델 가져오기
model_emotion = emotion()
model_mask = mask()
model_glasses = glasses()
model_sunglasses = sunglasses()

# 분류될 감정리스트
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# 대상 사진파일경로
file_path = './face/irene.png'

# 대상사진
image = cv2.imread(file_path)

# 모델별로 쓰일 이미지 변환
RGBimage, GRAYimage = trans_image_model(image)

result_mask = model_mask.predict(RGBimage)
result_glasses = model_glasses.predict(RGBimage)
print(file_path)
if result_mask[0][0] > result_mask[0][1]:
    print('해당 사진은 마스크를 썼습니다.')
else:
    # 마스크로 인해 얼굴인식이 안되서 마스크 판별이후 얼굴정렬을 이용한다.
    faceAligned = face_align(image)
    
    result_emotion = model_emotion.predict(GRAYimage)
    print('예측: ' + file_path)
    maxindex = int(np.argmax(result_emotion))
    print(emotion_dict[maxindex])
        
if result_glasses[0][0] < result_glasses[0][1]:
    print('해당 사진은 안경을 쓰지않았습니다.')
else:
    result_sunglasses = model_sunglasses.predict(RGBimage)
    if max(result_sunglasses[0]) == result_sunglasses[0][0]:
         print('해당 사진은 안경을 썼습니다.')
    else:
         print('해당 사진은 선글라스를 썼습니다.')
        
print('#####################################')

# 피부색      -> 피부색에 따라                     
# 마스크      -> no 입     -> mask              => 0 : 입 , 1: 마스크
#           -> no 감정표현 -> 눈 : netural      => default 를 netural로
# 안경       -> 안경추가                        => 0: 안경X , 1: 안경O
