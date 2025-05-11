# detect.py
import cv2
import numpy as np
from cv2 import ml

def get_hog_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    return np.append(sv.flatten(), -rho)

def main():
    # 1) HOG 설정 (학습과 동일)
    win_size     = (64, 128)
    block_size   = (16, 16)
    block_stride = (8, 8)
    cell_size    = (8, 8)
    nbins        = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # 2) 학습된 SVM 로드
    svm = ml.SVM_load('hog_thermal_svm.yml')
    hog.setSVMDetector(get_hog_detector(svm))

    # 3) 이미지 읽기 (PNG 컬러맵 포함 가능)
    orig = cv2.imread('test/image5.png')
    if orig is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 컬러맵→그레이스케일 (열화상 raw가 아닌 경우에도 여기서 처리)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # 4) detectMultiScale
    rects, weights = hog.detectMultiScale(
        gray,
        winStride=(8,8),
        padding=(8,8),
        scale=1.05
    )

    # 5) NMS(비최대 억제)
    boxes = [list(r) for r in rects]
    scores = [float(w) for w in weights]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.3)

    # 6) 결과 그리기
    for i in idxs:
        x, y, w, h = boxes[i[0]]
        cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(orig, 'Person', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Thermal HOG+SVM Detection', orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
