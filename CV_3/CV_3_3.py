import cv2 as cv#opencv라이브러리 불러오기
import matplotlib.pyplot as plt #Matplotlib를 사용하기위해 pyplot 모듈 불러오기
import numpy as np #NumPy 라이브러리 불러오기

#이미지 파일 불러오기
img = cv.imread("coffee cup.jpg")
#Matplotlib는 RGB를 사용하기 때문에 원본 이미지를 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#mask는 이미지와 같은 높이, 너비를 가짐
#각 픽셀이 배경인지 객체인지에 대한 정보를 저장하는 배열
#처음은 전부 0으로 초기화
mask = np.zeros(img.shape[:2], np.uint8)

#bgdModel : 배경 모델 저장용 배열
#fgdModel : 객체 모델 저장용 배열
#GrabCut 내부 알고리즘에서 사용되며 직접 값을 넣을 필요 없음
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

#객체가 포함된 대략적인 사각형 영역을 지정
#(x, y, w, h) 형태로 사각형의 왼쪽 상단 좌표와 너비, 높이를 지정
rect = (100, 80, 1000, 800)

#cv.grabCut(입력이미지, 배경/객체 정보를 저장할 마스크, 
# 초기 사각형영역, 배경모델, 객체모델, 반복횟수, 초기화 방법)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 확실한 배경(0) + 배경일 가능성이 큰 영역(2)은 0
# 나머지(객체 / 객체일 가능성이 큰 영역)는 1
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype("uint8")

#객체 영역은 유지되고, 배경 영역은 제거된 결과 이미지 생성
result = img_rgb * mask2[:, :, np.newaxis]

#mask는 0과 1로 이루어진 이진 이미지이므로 시각화를 위해 255를 곱하여 흑백으로 표시
#배경은 검정(0), 객체는 흰색(255)으로 표시
mask_vis = mask2 * 255

#원본 RGB 이미지를 복사하여 사각형을 그릴 별도 이미지 생성
img_rect = img_rgb.copy()
x, y, w, h = rect #rect에서 사각형의 위치와 크기를 추출
#초기 사각형 영역을 파란색으로 그림
#cv.rectangle(이미지, 시작점, 끝점, 색상, 두께)
cv.rectangle(img_rect, (x, y), (x + w, y + h), (255, 0, 0), 2)

#전체 출력 화면 크기 설정
plt.figure(figsize=(15, 5)) 

plt.subplot(1,3,1) #출력 화면을 1행3열로 나눴을때 첫번째 위치에 원본 이미지 표시
plt.imshow(img_rect) #원본 이미지 출력
plt.title("Original") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,3,2) #출력 화면을 1행3열로 나눴을때 두번째 위치에 에지 강도 이미지 표시
plt.imshow(mask_vis, cmap="gray") #에지 강도 이미지를 그레이스케일로 출력
plt.title("Mask Image") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

plt.subplot(1,3,3) #출력 화면을 1행3열로 나눴을때 세번째 위치에 에지 강도 이미지 표시
plt.imshow(result) #에지 강도 이미지를 그레이스케일로 출력
plt.title("Background Removed") #제목 표시
plt.axis("off") #깔끔한 표시를 위해 축 제거

#subplot 간 간격을 자동으로 조절하여 겹치지 않게 함
plt.tight_layout()

#화면에 출력
plt.show()