import cv2
import numpy as np

orig = cv2.imread('test/image3.png')
if orig is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# detectMultiScale로 사람 검출
boxes, weights = hog.detectMultiScale(orig, winStride=(8,8), padding=(8,8), scale=1.05)

for (x, y, w, h) in boxes:
    cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(orig, 'Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.imshow('People Detection', orig)
cv2.waitKey(0)
cv2.destroyAllWindows()


