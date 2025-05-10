import cv2
import numpy as np

orig = cv2.imread('test/test.png')
if orig is None:
    print("이미지를 불러올 수 없습니다.")
    exit()
h, w = orig.shape[:2]

# 1. 강한 블러+Morphology로 노이즈 제거
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (13, 13), 5)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphology closing으로 구멍/틈 메우기
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 외곽선 검출
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Hull strong contours:", len(contours))

# (디버깅) 모든 컨투어 면적 출력
for i, cnt in enumerate(contours):
    print(f"{i}: area={cv2.contourArea(cnt)}")

# 가장 큰 것만 (min_area로 과감히 필터링)
filtered = [c for c in contours if cv2.contourArea(c) > (h*w)*0.18]
if filtered:
    best = max(filtered, key=cv2.contourArea)
    disp = orig.copy()
    cv2.drawContours(disp, [best], -1, (0,0,255), 5)
    x, y, ww, hh = cv2.boundingRect(best)
    cv2.putText(disp, 'Hull', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Hull only', disp)
    cv2.waitKey(0)
else:
    print("No valid hull found!")


# 2. 내부 구조(함포/함교) 검출용 이미지 복제: 샤프닝+대비 강조
img_feat = orig.copy()
gray_feat = cv2.cvtColor(img_feat, cv2.COLOR_BGR2GRAY)
gray_feat = cv2.GaussianBlur(gray_feat, (3, 3), 0)
gray_feat = cv2.equalizeHist(gray_feat)
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gray_feat = cv2.filter2D(gray_feat, -1, sharpen_kernel)

edges_feat = cv2.Canny(gray_feat, 10, 50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
edges_feat = cv2.dilate(edges_feat, kernel, iterations=1)
edges_feat = cv2.erode(edges_feat, kernel, iterations=1)
contours_feat, _ = cv2.findContours(edges_feat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




# --- Turret (함포): HoughCircles + 컨투어 병행
turret_img = orig.copy()
turret_candidates = []
min_area = (h * w) * 0.005
max_area = (h * w) * 0.2
for cnt in contours_feat:
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        continue
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x, y, ww, hh = cv2.boundingRect(approx)
    cx, cy = x + ww // 2, y + hh // 2
    aspect = ww / hh if hh != 0 else 0
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    fill_ratio = area / circle_area if circle_area > 0 else 0
    if fill_ratio > 0.5 and 0.7 < aspect < 1.3:
        if cx < w * 0.33 or cx > w * 0.67:
            turret_candidates.append((cx, cy, int(radius)))

circles = cv2.HoughCircles(
    gray_feat, cv2.HOUGH_GRADIENT, dp=1.3, minDist=50,
    param1=60, param2=36, minRadius=18, maxRadius=38)
if circles is not None:
    circles = np.uint16(np.around(circles[0]))
    for (x, y, r) in circles:
        if (x < w*0.33 or x > w*0.67) and all(np.hypot(x-cx, y-cy) > 30 for (cx, cy, cr) in turret_candidates):
            turret_candidates.append((x, y, r))

# 각 쪽 최대 1개만
left = [c for c in turret_candidates if c[0] < w*0.33]
right = [c for c in turret_candidates if c[0] > w*0.67]
if left:
    lmost = max(left, key=lambda x: x[2])
    cv2.circle(disp, (lmost[0], lmost[1]), lmost[2], (0,255,255), 3)
    cv2.putText(disp, 'Turret', (lmost[0], lmost[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
if right:
    rmost = max(right, key=lambda x: x[2])
    cv2.circle(disp, (rmost[0], rmost[1]), rmost[2], (0,255,255), 3)
    cv2.putText(disp, 'Turret', (rmost[0], rmost[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

# --- Bridge(함교): 중앙 40%에 위치한 사각형
for cnt in contours_feat:
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        continue
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x, y, ww, hh = cv2.boundingRect(approx)
    cx, cy = x + ww // 2, y + hh // 2
    aspect = ww / hh if hh != 0 else 0
    if 3 <= len(approx) <= 8 and 0.4 < aspect < 2:
        if w * 0.3 < cx < w * 0.7:
            cv2.drawContours(disp, [approx], -1, (0, 255, 0), 4)
            cv2.putText(disp, 'Bridge', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow('Turrets and Bridge', disp)
cv2.imwrite('result_final.png', disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

