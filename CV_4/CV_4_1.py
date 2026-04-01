import cv2 as cv #opencv라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기

#이미지 파일 불러오기(BGR 형식)
org = cv.imread("mot_color70.jpg")
gray = cv.cvtColor(org, cv.COLOR_BGR2GRAY) #이미지 파일을 그레이스케일로 변환

sift=cv.SIFT_create(nfeatures=200) #SIFT 객체 생성, 최대 200개의 키포인트를 검출하도록 설정
kp, des = sift.detectAndCompute(gray, None) #SIFT 알고리즘

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #키포인트를 그레이스케일 이미지에 그리기


#Matplotlib는 RGB를 사용하기 때문에 원본 이미지를 RGB로 변환
org = cv.cvtColor(org, cv.COLOR_BGR2RGB) 

#전체 출력 화면 크기 설정
plt.figure(figsize=(10, 5)) 

plt.subplot(1,2,1) #출력 화면을 1행2열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(org) #원본 이미지 출력
plt.title("Original Image") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,2,2) #출력 화면을 1행2열로 나눴을때 두번째 위치에 에지 강도 이미지 표시
plt.imshow(gray, cmap='gray') #에지 강도 이미지를 그레이스케일로 출력
plt.title("SIFT Keypoints") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()