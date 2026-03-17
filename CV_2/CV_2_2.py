import cv2 #opencv라이브러리 불러오기

#이미지 불러오기
img_org = cv2.imread("L02_lab/images/rose.png")
#결과를 보기 쉽게 원본 사진의 크기를 절반으로 줄임
img = cv2.resize(img_org, None, fx=0.5, fy=0.5)

# 이미지를 불러오기 실패한 경우 출력
if img is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

#이미지 크기 구하기
#img.shape은 (height, width, channels) 형태이므로 앞의 두 값만 가져옴
#height와 width에 세로, 가로 크기 저장
height, width = img.shape[:2] 

#회전을 할 때 기준이 되는 중심 좌표 계산
#center는 이미지 중앙 좌표를 나타냄
center = (width // 2, height // 2)

# cv2.getRotationMatrix2D(중심좌표, 회전각도, 크기비율)
# 중심 기준으로 30도 회전, 크기는 0.8배
M = cv2.getRotationMatrix2D(center, 30, 0.8)


# x축으로 +80, y축으로 -40 평행이동 값 추가
# x좌표와 y좌표의 이동값을 변경시켜줌
M[0, 2] += 80
M[1, 2] += -40

# cv2.warpAffine(원본 이미지, 변환 행렬 M, 결과 이미지 크기)
# 위에서 만든 변환 행렬 M을 이용해 실제 이미지에 변환 적용
result = cv2.warpAffine(img, M, (width, height))

cv2.imshow("Original", img) # 원본 이미지 출력
cv2.imshow("Rotated + Scaled + Translated", result) # 변환된 이미지 출력

cv2.waitKey(0) # 키 입력이 들어올 때까지 창을 유지
cv2.destroyAllWindows() # 모든 opencv 창 닫기
