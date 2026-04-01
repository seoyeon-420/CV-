import cv2 as cv #opencv라이브러리 불러오기
import numpy as np #NumPy 라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기

#이미지 불러오기(BGR 형식)
img1 = cv.imread("img1.jpg")
img2 = cv.imread("img2.jpg")
#이미지를 그레이스케일로 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create() #SIFT 객체 생성
#각 이미지에서 특징점(kr)과 디스크립터(descriptor) 계산
kp1, des1 = sift.detectAndCompute(gray1, None) 
kp2, des2 = sift.detectAndCompute(gray2, None) 

bf = cv.BFMatcher(cv.NORM_L2) #BFMatcher 객체 생성
#각 특징점에 대해 가장 가까운 2개의 매칭 후보를 찾음
matches = bf.knnMatch(des1, des2, k=2) 

good_matches = [] #매칭점 선별을 위한 리스트 생성
for m, n in matches:     
    if m.distance < 0.7 * n.distance: 
    #두 개의 최근접 이웃의 거리 비율이 임계값 0.7 미만인 매칭점만 선별
        good_matches.append(m)

# distance가 작은 순서대로 정렬하여 상위 50개 매칭점만 사용(노이즈 증가 방지)
good_matches = sorted(good_matches, key=lambda x: x.distance)
good_matches = good_matches[:50]

#최소 4개 이상의 매칭점이 있어야 호모그래피 계산 가능
#계산 가능 여부 확인
if len(good_matches) < 4:
    raise ValueError("좋은 매칭점이 부족합니다.")

#img1의 매칭점 좌표
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#img2의 대응점 좌표
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#두 이미지 사이의 변환 행렬 H 계산
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# img2를 img1 좌표계로 가져오기 위해 역행렬 계산
H_inv = np.linalg.inv(H)

# 이미지 정합을 위한 캔버스 크기 계산
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# img2를 변형해서 큰 캔버스에 배치
warped = cv.warpPerspective(img2, H_inv, (w1 + w2, max(h1, h2)))

# img1을 기준 이미지로 왼쪽에 그대로 배치
warped[0:h1, 0:w1] = img1

# 두 이미지의 매칭 결과를 선으로 연결하여 시각화
matching_result = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Matplotlib는 RGB를 사용하기 때문에 이미지를 RGB로 변환
matching_result = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)
warped = cv.cvtColor(warped, cv.COLOR_BGR2RGB)


plt.figure(figsize=(16, 7)) #전체 출력 창 크기 설정

plt.subplot(1,2,1) #출력 화면을 1행2열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(matching_result) #원본 이미지 출력
plt.title("Original Image") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,2,2) #출력 화면을 1행2열로 나눴을때 두번째 위치에 에지 강도 이미지 표시
plt.imshow(warped) #에지 강도 이미지를 그레이스케일로 출력
plt.title("Warped Image") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()