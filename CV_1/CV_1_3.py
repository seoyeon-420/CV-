import cv2 as cv
import sys

img = cv.imread('soccer.jpg') #이미지를 읽어와 img변수에 저장

#이미지 파일을 찾지 못했을 때 오류 메시지 출력 후 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다')

original = img.copy() #원본 보호를 위해 복사하여 original변수에 저장

drawing = False #현재 마우스를 드래그 중인지 여부를 저장하는 변수
selecting = True #ROI 선택 가능 상태
ix, iy = -1, -1 #드래그 시작 좌표를 저장할 변수
roi = None #잘라낸 ROI이미지를 저장할 변수
save_count = 0 #저장할 파일 이름 번호를 위한 카운트

def draw(event, x, y, flags, param): #마우스 이벤트가 발생할 때 실행되는 함수 정의
    global ix, iy, drawing, roi, selecting #함수 내부에서 전역변수 사용

    if not selecting: #ROI창이 떠있으면 마우스를 무시하고 함수 종료
        return

    if event == cv.EVENT_LBUTTONDOWN: #마우스 왼쪽 버튼을 눌렀을 때
        drawing = True #드래그 시작 상태로 변경
        ix, iy = x, y #현재 마우스 좌표를 시작점으로 저장

    elif event == cv.EVENT_MOUSEMOVE: #마우스를 움직일때
        if drawing: #드래그 중
            temp = original.copy() #원본 이미지를 복사해서 temp생성
            cv.rectangle(temp, (ix, iy), (x, y), (0,0,255), 2) #드래그 영역에 계속 사각형 생성
            cv.imshow('Drawing', temp) #사각형이 그려진 임시 이미지를 화면에 표시

    elif event == cv.EVENT_LBUTTONUP: #마우스 왼쪽 버튼을 뗐을 때
        drawing = False #드래그 종료
        selecting = False #ROI창 뜨는 동안 선택 금지

        x1 = min(ix, x)
        x2 = max(ix, x)
        y1 = min(iy, y)
        y2 = max(iy, y)

        temp = original.copy() #원본 이미지를 다시 복사
        cv.rectangle(temp, (x1, y1), (x2, y2), (0,0,255), 2) #선택된 영역을 사각형으로 표시
        cv.imshow('Drawing', temp) #사각형이 그려진 이미지 표시

        roi = original[y1:y2, x1:x2] #Numpy 슬라이싱으로 ROI영역 잘라내기
        cv.imshow('new', roi) #잘라낸 영역을 새 창에 표시

cv.namedWindow('Drawing') #Drawing이라는 이름의 창 생성
cv.imshow('Drawing', img) #원본 이미지를 Drawing 창에 표시

cv.setMouseCallback('Drawing', draw) #Drawing창에서 마우스 이벤트가 발생하면 draw함수 실행

while True: #프로그램이 계속 실행되도록 무한 반복
    key = cv.waitKey(1) #키 입력을 1ms동안 기다림

    if key == ord('q'):  #'q'키를 누르면 창을 닫고 반복문을 종료하여 프로그램을 끝냄
        cv.destroyAllWindows()
        break

    elif key == ord('r') and not selecting: #r키를 눌렀고 ROI창이 떠있는 상태라면
        cv.destroyWindow('new') #새로 만들어진 new창 닫기
        cv.imshow('Drawing', original) #사각형이 사라진 원본 이미지 다시 표시
        selecting = True #다시 ROI 선택 가능 상태로 변경

    elif key == ord('s') and not selecting: #s키를 눌렀고 ROI창이 떠있는 상태라면
        filename = f'roi_{save_count}.png' #저장할 파일 이름 생성
        cv.imwrite(filename, roi) #ROI 이미지를 파일로 저장
        print(filename, "저장 완료") #저장 완료 메시지 출력
        save_count += 1 #다음 저장을 위해 번호 증가