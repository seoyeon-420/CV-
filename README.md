# Computer Vision



## 실습1_1 이미지불러오기및그레이스케일변환

- OpenCV를 사용하여 이미지를 불러오고 화면에 출력하기
- 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시하기

### 요구사항
1. cv.imread()를 사용하여 이미지 로드
2. cv.cvtColor함수를 사용해 이미지를 그레이스케일로 변환
3. np.hstack()함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
4. cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무키나 누르면 창이 닫히도록 할 것

### 전체 코드
```python
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
```
### 고려 사항
```phthon
gray_small=cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)
```
np.hstak()을 이용하여 이미지를 연결하기 위해서는 차원이 같아야 하기때문에 흑백으로 바꿔 2차원이된 색을 3차원으로 바꾸어준다.

## 실습1_2 페인팅 붓 크기 조절 기능 추가

- 마우스 입력으로 이미지 위에 붓질
- 키보드 입력을 이용해 붓의 크기를 조절하는 기능 추가

### 요구사항
1. 초기 붓 크기는 5를 사용
2. +입력 시 붓 크기 1증가, -입력 시 붓 크기 1감소
3. 붓 크기는 최소1, 최대 15로 제한
4. 좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기
5. q키를 누르면 영상 창이 종료

### 전체 코드
```python
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
```

### 고려사항
```python
key = cv.waitKey(1)
```
키보드 이벤트는 3가지지만 cv.waitKey(1)를 여러번 사용 시 키가 먹히는 현상이 발생하므로 하나의 변수로 묶어주어 사용한다.

## 실습1_3 마우스로 영역 선택 및 ROI(관심영역) 추출

-이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택
-선택한 영역만 따로 저장하거나 표시

### 요구사항
1. 이미지를 불러오고 화면에 출력
2. cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
3. 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택
4. 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
5. r키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
6. s키를 누르면 선택한 영역을 이미지 파일로 저장

### 전체코드
```python
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
```
