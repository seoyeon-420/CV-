import cv2 as cv#opencv라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기
import numpy as np #NumPy 라이브러리 불러오기

#이미지 파일 불러오기
org = cv.imread("dabo.jpg")
#원본 이미지를 복사하여 결과 이미지로 사용
result = org.copy() 
#그레이 스케일 변환
gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

#Canny 알고리즘을 이용해 에지 검출 (낮은 임계값: 100, 높은 임계값: 200)
#값이 클수록 강한 에지만 검출됨
canny=cv.Canny(gray, 100, 200)

#직선 검출
#cv.HoughLinesP(입력이미지, 거리 해상도, 각도 해상도, 임계값, 최소 선 길이, 최대 선 간격)
lines=cv.HoughLinesP(canny,1, np.pi/180, threshold=120, minLineLength=50, maxLineGap=10) #허프 변환을 이용한 선 검출

#검출된 직선을 이미지에 그리기
#lines: 검출된 선들의 배열
for line in lines:
    x1, y1, x2, y2 = line[0] #각 직선은 (x1, y1, x2, y2) 형태로 저장되어 있음

    #cv.line(이미지, 시작점, 끝점, 색상, 두께) 
    #BGR 형식으로 색을 지정하여 (0, 0, 255)가 빨간색을 의미
    cv.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2) 

org = cv.cvtColor(org, cv.COLOR_BGR2RGB) #Matplotlib는 RGB를 사용하기 때문에 원본 이미지를 RGB로 변환
result = cv.cvtColor(result, cv.COLOR_BGR2RGB) #결과 이미지도 RGB로 변환

#전체 출력 화면 크기 설정
plt.figure(figsize=(10, 5)) 

plt.subplot(1,2,1) #출력 화면을 1행2열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(org) #원본 이미지 출력
plt.title("Original") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,2,2) #출력 화면을 1행2열로 나눴을때 두번째 위치에 에지 강도 이미지 표시
plt.imshow(result) #에지 강도 이미지를 그레이스케일로 출력
plt.title("line") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()