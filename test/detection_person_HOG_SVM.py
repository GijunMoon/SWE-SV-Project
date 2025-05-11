import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 감마 보정
    invGamma = 1.0 / 1.2
    table = np.array([((i/255.0)**invGamma)*255 for i in range(256)], dtype="uint8")
    gray = cv2.LUT(gray, table)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def detect_people(orig):
    gray = preprocess(orig)

    # 2× 업샘플링
    scale = 2.0
    large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # HOG + 기본 PeopleDetector 로드
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ✱ 포지셔널 인자만 전달: (winStride, padding, scale)
    rects, weights = hog.detectMultiScale(
        large,
        0.0,           # hitThreshold
        (4, 4),        # winStride
        (8, 8),        # padding
        1.02           # scale
    )


    # 원본 크기로 좌표 복원 및 score 수집
    boxes, scores = [], []
    for (x, y, w, h), score in zip(rects, weights):
        boxes.append([int(x/scale), int(y/scale), int(w/scale), int(h/scale)])
        scores.append(float(score))

    # NMS 적용
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.6, 0.4)
    return [boxes[i[0]] for i in idxs] if len(idxs) > 0 else []

def draw_boxes(img, boxes):
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, 'Person', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

if __name__ == "__main__":
    orig = cv2.imread('test/image5.png')
    if orig is None:
        print("이미지를 불러올 수 없습니다.")
        exit()

    boxes = detect_people(orig)
    draw_boxes(orig, boxes)

    cv2.imshow('Result', orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
