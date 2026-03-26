# Computer Vision

# CV4_Local Feature실습


## 실습4_1 SIFT를 이용한 특징점 검출 및 시각화
- 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform)알고리즘을 사용하여 특징점을 검출하고 이를 시각화

### 요구사항
1. cv.SIFT_create()를 사용하여 SIFT 객체를 생성
2. detectAndCompute()를 사용하여 특징점을 검출
3. cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
4. matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력

### 전체 코드
```python
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
```
### 결과 이미지
![결과1](CV_4_1_result.png)

### 기억사항
```python
sift=cv.SIFT_create(nfeatures=200)
kp, des = sift.detectAndCompute(gray, None) 
gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
```
cv.SIFT_create(nfeatures=200)를 사용하여 SIFT 객체를 생성, nfeatures를 사용해 최대 200개의 특징점을 검출하도록 제한을 둠  
detectAndCompute()를 통해 이미지에서 특징점(keypoint)과 descriptor를 추출함  
cv.drawKeypoints()를 사용하여 이미지 위에 특징점을 시각적으로 표시함  
DRAW_RICH_KEYPOINTS 옵션을 이용해 특징점의 크기, 방향 정보까지 담음

## 실습4_2 SIFT를 이용한 두 영상 간 특징점 매칭
- 두 개의 이미지(mot_color70.jpg,mot_color80.jpg)를 입력받아 SIFT특징점 기반으로 매칭을 수행하고 시각화

### 요구사항
1. cv.imread()를 사용하여 두 개의 이미지를 불러옴
2. cv.SIFT_create()를 사용하여 특징점을 추출
3. cv.BFMatcher()또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
4. cv.drawMatches를 사용하여 매칭 결과를 시각화
5. matplotlib을 이용하여 매칭 결과를 출력

### 전체 코드
```python
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
```

### 결과 이미지
![결과2](CV_4_2_result.png)

### 기억사항
```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True) #BFMatcher 생성
```
NORM_L2 : SIFT descriptor는 실수형이므로 L2 거리 사용  
crossCheck=True : 서로 가장 잘 맞는 경우만 매칭 (더 정확하지만 매칭 수는 줄어듦)

```python
matches = bf.match(des1, des2)
```
des1과 des2를 비교하여 가장 비슷한 특징점끼리 연결함

```python
matched_img = cv.drawMatches(img1, kp1,img2, kp2,
    matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
매칭된 점들을 시각화  
(점을 찍을 이미지1, 첫이미지의 특징점리스트, 시각화할 두번째 이미지, 두번째이미지의 특징점리스트,  
사용할 매칭 결과, 이미지 저장 공간, 매칭되지 않은 것은 그리지 않기)


## 실습4_3 호모그래피를 이용한 이미지 정합(Image Alignment)
- SIFT특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
- 샘플파일로 img1.jpg, img2.jpg, img3.jpg 중 2개를 선택

### 요구사항
1. cv.imread()를 사용하여 두 개의 이미지를 불러옴
2. cv.SIFT_creat()를 사용하여 특징점을 검출
3. cv.BFMatcher와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
4. cv.findHomography()를 사용하여 호모그래피 행렬을 계산
5. cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
6. 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력

### 전체코드
```python

```

### 결과 이미지
![결과3](CV_4_3_result.png))

### 기억사항
```python

```
