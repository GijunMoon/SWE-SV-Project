# train.py
import os
import cv2
import numpy as np
from cv2 import ml

def load_hog_samples(dir_path, label, hog, win_size):
    feats, labs = [], []
    for fn in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, fn), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, win_size)
        vec = hog.compute(img)
        feats.append(vec.flatten())
        labs.append(label)
    return feats, labs

def main():
    # 1) HOG 파라미터 설정 (Dalal & Triggs)
    win_size     = (64, 128)
    block_size   = (16, 16)
    block_stride = (8, 8)
    cell_size    = (8, 8)
    nbins        = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # 2) 데이터 로드
    pos_feats, pos_labs = load_hog_samples('train/data/pos', +1, hog, win_size)
    neg_feats, neg_labs = load_hog_samples('train/data/neg', -1, hog, win_size)

    X = np.array(pos_feats + neg_feats, dtype=np.float32)
    y = np.array(pos_labs + neg_labs, dtype=np.int32)

    # 3) SVM 학습
    svm = ml.SVM_create()
    svm.setType(ml.SVM_C_SVC)
    svm.setKernel(ml.SVM_LINEAR)
    svm.setC(0.01)
    svm.train(X, ml.ROW_SAMPLE, y)
    svm.save('hog_thermal_svm.yml')
    print("학습 완료: hog_thermal_svm.yml 저장")

if __name__ == '__main__':
    main()
