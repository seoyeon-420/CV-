import cv2 as cv #opencv라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기

#이미지 파일 불러오기(BGR 형식)
img1 = cv.imread("mot_color70.jpg")
img2 = cv.imread("mot_color83.jpg")
#이미지 파일을 그레이스케일로 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create() #SIFT 객체 생성

kp1, des1 = sift.detectAndCompute(gray1, None) #SIFT 알고리즘
kp2, des2 = sift.detectAndCompute(gray2, None) #SIFT 알고리즘

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)#BFMatcher 생성

# 두 이미지의 descriptor를 매칭
matches = bf.match(des1, des2)

# 거리(distance)가 작은 순서대로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 50개 매칭만 시각화
matched_img = cv.drawMatches(img1, kp1,img2, kp2,
    matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Matplotlib는 RGB를 사용하기 때문에 원본 이미지를 RGB로 변환
matched_img= cv.cvtColor(matched_img, cv.COLOR_BGR2RGB) 

#전체 출력 화면 크기 설정
plt.figure(figsize=(10, 5)) 

plt.subplot(1,1,1) #출력 화면을 1행1열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(matched_img) #원본 이미지 출력
plt.title("matched_img") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()