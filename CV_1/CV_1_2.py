import cv2 as cv #open cv 라이브러리 사용
import sys

img = cv.imread('soccer.jpg') #사진 불러오기

#이미지 파일을 찾지 못했을 때 오류 메시지 출력 후 프로그램 종료
if img is None :
    sys.exit('파일을 찾을 수 없습니다')

pen=5 #펜의 기본 굵기를 5로 설정

def draw(event,x,y,flags,param): #마우스 이벤트가 발생할 때 실행되는 함수 정의
    if event == cv.EVENT_MOUSEMOVE: #마우스를 움직였을 때 발생하는 이벤트
        if flags & cv.EVENT_FLAG_LBUTTON: #마우스의 왼쪽 버튼을 누른 상태에서 이동중 발생하는 이벤트
            cv.circle(img,(x,y),pen,(225,0,0),-1) #

        elif flags & cv.EVENT_FLAG_RBUTTON: #마우스의 오른쪽 버튼을 누른 상태에서 이동중 발생하는 이벤트
            cv.circle(img,(x,y),pen,(0,0,225),-1)

    cv.imshow('Drawing',img) #Drawing 이라는 창에 현재 이미지를 다시 표시하여 그림이 보이게 함

cv.namedWindow('Drawing') #'Drawing창 생성
cv.imshow('Drawing', img) #원본 이미지를 생성된 창에 표시

cv.setMouseCallback('Drawing', draw) #'Drawing이라는 창에서 마우스 이벤트 발생 시 draw함수 실행

while(True):
    key = cv.waitKey(1) #=>if문마다 선언하던 것을 하나의 변수로 바꾸어 사용함.

    if key==ord('='): #'=(+)'키를 누를 시 굵기 증가'
        if(pen<15): #펜굵기가 15를 못넘도록 제한
            pen+=1
    if key==ord('-'): #'-'키를 누를 시 굵기 감소'
        if(pen>1): #펜굵기가 1보다 작아지지 못하도록 제한
            pen-=1

    if key==ord('q'): #'q'키를 누르면 창을 닫고 반복문을 종료하여 프로그램을 끝냄
        cv.destroyAllWindows()
        break

