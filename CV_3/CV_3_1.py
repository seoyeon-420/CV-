import cv2 as cv #opencv라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기

#이미지 파일 불러오기(BGR 형식)
org = cv.imread("edgeDetectionImage.jpg")
#그레이 스케일 변환
gray = cv.cvtColor(org, cv.COLOR_BGR2GRAY)

#Sobel 필터를 사용하여 x방향, y방향 각각 에지 검출
gray_x=cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) # (1,0)-> x방향 미분
gray_y=cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3) # (0,1)-> y방향 미분
#cv.Sobel(입력이미지, 출력 데이터 타입, dx, dy, 커널 크기)
#64비트 부호 있는 실수형(F)으로 결과를 저장하여 음수값도 표현할 수 있도록 함

#x, y방향 에지 값을 이용하여 전체 에지 강도 계산
edge_strength=cv.magnitude(gray_x, gray_y) #에지 강도 계산

#계산된 에지 강도는 실수형이므로 시각화를 위해 절댓값을 취하고 uint8로 변환
edge_strength = cv.convertScaleAbs(edge_strength)

#Matplotlib는 RGB를 사용하기 때문에 원본 이미지를 RGB로 변환
org = cv.cvtColor(org, cv.COLOR_BGR2RGB) 

#전체 출력 화면 크기 설정
plt.figure(figsize=(10, 5)) 

plt.subplot(1,2,1) #출력 화면을 1행2열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(org) #원본 이미지 출력
plt.title("Original Image") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,2,2) #출력 화면을 1행2열로 나눴을때 두번째 위치에 에지 강도 이미지 표시
plt.imshow(edge_strength, cmap='gray') #에지 강도 이미지를 그레이스케일로 출력
plt.title("Edge Strength") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()
