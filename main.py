import cv2
from emoticon import make_emoticon

# 대상 사진파일경로
file_path = './face/me.png'
file_path = './masked/glasses/mask_0902.jpg'

# 대상사진
image = cv2.imread(file_path)

face, code = make_emoticon(image)
# 결과출력
cv2.imwrite('result' + '.png',face)