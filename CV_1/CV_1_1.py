import cv2 as cv #open cv 라이브러리 사용
import sys
import numpy as np #np.hstack을 사용하기 위함

img = cv.imread('soccer.jpg') #사진 불러오기

if img is None :
    sys.exit('파일이 존재하지 않습니다.')

img_small=cv.resize(img,dsize=(0,0), fx=0.5, fy=0.5)#사진 반으로 축소
gray_small=cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)#축소된 사진 흑백버전
gray_small=cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)#흑백사진은 2차원 컬러이기 때문에 합쳐주기 위해 3차원 컬러로 전환

answer=np.hstack((img_small, gray_small)) #두 사진을 하나의 창으로 나타냄

cv.imshow('answer', answer) #합쳐진 이미지 출력

cv.waitKey() # 아무 키나 입력하여 창이 닫히도록함
cv.destroyAllWindows()
