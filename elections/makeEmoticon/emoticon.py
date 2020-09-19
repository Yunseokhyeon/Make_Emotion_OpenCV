import numpy as np
import tensorflow as tf
import cv2
from .model import *
from .util import trans_image_model, reg_face, add_face_part
import os.path

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# 학습된 모델 가져오기
model_emotion = ''
model_mask = ''
model_skinGlasses = ''
model_sunglasses = ''
model_roundrec = ''


# 분류될 감정리스트
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 
                3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
skin_dict = {0: "asian", 1:"black", 2:"white"}
mask_dict = {0: "mask", 1:"noMask"}
glasses_dict = {0: "glasses", 1:"noGlasses"}
glasses_shape_dict = {0: "rectangle", 1:"round", 2:"sunglasses"}

# 메인로직
def make_emoticon(image):
    result={}
    # 초기 설정 마스크를 꼈을 경우 default Value
    # skin = asian / emotion = neutral
    result['skin'] = skin_dict[0]
    result['emotion'] = emotion_dict[4]
    result['glasses'] = glasses_dict[1]
    result['glasses_shape'] = []

    # 모델별로 쓰일 이미지 변환
    RGBimage, GRAYimage = trans_image_model(image)
    
    cv2.imshow("m", RGBimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 마스크 유무 판별
    model_mask = mask()
    result_mask = model_mask.predict(RGBimage)
    print("-----------------------")
    print("마스크 모델확인")
    print(result_mask)
    print(result_mask[0])
    print(np.argmax(result_mask[0]))
    result['mask'] = mask_dict[np.argmax(result_mask[0])]
    
    code = 0

    # 마스크를 끼지 않은 경우 피부색과 감정인식을 할 수 있다.
    if result['mask'] == 'noMask' :
        image, code = reg_face(image)

        # code  0 : 얼굴인식 완료
        #      -1 : 얼굴다수 인식
        #      -2 : 얼굴인식 실패
        if code == -1:
            print("사람이 1명 이상입니다.")
        elif code == -2:
            print("사람을 찾지 못하였습니다")
        else :
            # 얼굴인식한 이미지로 다시 이미지 변환
            RGBimage, GRAYimage = trans_image_model(image)

            # 결과를 저장할 dictionary

            
            ###### 피부색 판별 ######
            # 0: 아시안 + 안경O     #
            # 1: 아시안 + 안경X     #
            # 2: 흑인 + 안경O      #
            # 3: 흑인 + 안경X      #
            # 4: 백인 + 안경O      #
            # 5: 백인 + 안경X      #
            ######################
            model_skinGlasses = skinGlasses()
            result_skin = model_skinGlasses.predict(RGBimage)

            maxSkinGlasses = np.argmax(result_skin[0])
            result['skin'] = skin_dict[int(maxSkinGlasses/2)]
            result['glasses'] = glasses_dict[int(maxSkinGlasses%2)]
            print("-------------------")
            print("피부 안경 모델 확인")
            print(result_skin)
            print("maxSkinGlasses = " + str(maxSkinGlasses))
            print("skin -> ", str(int(maxSkinGlasses/2)) )
            print("glasses -> ", str(int(maxSkinGlasses%2)) )
            print("-------------------")
        
            # 안경모델의 결과는 0:안경, 1:안경X
            # 선글라스모델의 결과는 0:선글라스X, 1:선글라스O
            # 안경을 끼고 있는경우 선글라스 모델사용
            if int(maxSkinGlasses%2) == 0:
                model_sunglasses = nomask_sunglasses()
                result_sunglasses = model_sunglasses.predict(RGBimage)
                maxSunglasses = int(np.argmax(result_sunglasses[0]))
                print("-------------------")
                print("노마스크 선글라스 모델확인")
                print(result_sunglasses)
                print(maxSunglasses)
                # 선글라스인 경우
                if maxSunglasses%2 == 1:
                    result['glasses_shape'] = glasses_shape_dict[2]
                # 선글라스가 아닌경우
                else :
                    # 흑인인경우
                    if maxSkinGlasses == 2 and maxSkinGlasses == 3:
                        model_roundrec = nomask_roundrec_black()
                    else:
                        model_roundrec = nomask_roundrec()
                    
                    result_roundrec = model_roundrec.predict(RGBimage)
                    maxRoundRec = np.argmax(result_roundrec[0])
                    result['glasses_shape'] = glasses_shape_dict[maxRoundRec]

                    print("-------------------")
                    print("노마스크 안경모양 모델확인")
                    print(result_roundrec)
                    print(maxRoundRec)

            # 감정인식
            model_emotion = emotion()
            result_emotion = model_emotion.predict(GRAYimage)
            result['emotion'] = emotion_dict[np.argmax(result_emotion)]
            print("-------------------")
            print("감정인식 모델확인")
            print(result_emotion)
            print(np.argmax(result_emotion))
    else:
        
        model_skinGlasses = mask_glasses()
        result_skin = model_skinGlasses.predict(RGBimage)

        maxSkinGlasses = np.argmax(result_skin[0])
        result['glasses'] = glasses_dict[int(maxSkinGlasses%2)]
        
        print("-------------------")
        print("마스크착용 안경모델확인")
        print(result_skin)
        print(maxSkinGlasses)

        # 안경모델의 결과는 0:안경, 1:안경X
        # 선글라스모델의 결과는 0:선글라스X, 1:선글라스O
        # 안경을 끼고 있는경우 선글라스 모델사용
        if int(maxSkinGlasses%2) == 0:
            model_sunglasses = mask_sunglasses()
            result_sunglasses = model_sunglasses.predict(RGBimage)

            maxSunglasses = np.argmax(result_sunglasses[0])

            print("-------------------")
            print("마스크착용 선글라스모델확인")
            print(result_sunglasses)
            print(maxSunglasses)

            # 선글라스인 경우
            if maxSunglasses%2 == 1:
                result['glasses_shape'] = glasses_shape_dict[2]
            # 선글라스가 아닌경우
            else :
                model_roundrec = mask_roundrec()
                    
                result_roundrec = model_roundrec.predict(RGBimage)
                maxRoundRec = np.argmax(result_roundrec[0])
                result['glasses_shape'] = glasses_shape_dict[maxRoundRec]

                print("-------------------")
                print("마스크착용 안경모양모델확인")
                print(result_roundrec)
                print(maxRoundRec)


    print(result)
    print('#####################################')

    # 이모티콘 생성영역
    face = cv2.imread(BASE_PATH + '/emoticon/skin/'+ result['skin'] +'.png')
    add_face_part(face, result['emotion'], 'eye', 130, 170) # 눈붙이기

    # 마스크 착용인 경우 입 대신 마스크를 착용한다.
    if result['mask'] == 'mask':
        add_face_part(face,'mask', 'white', 50, 290)
    else :
        add_face_part(face, result['emotion'], 'mouth', 180, 350 + 90) # 입붙이기

    # 안경 착용
    if result['glasses'] != 'noGlasses':
        add_face_part(face, 'glasses', result['glasses_shape'], 110, 120) #안경

    image = face
    

    
    return image, code