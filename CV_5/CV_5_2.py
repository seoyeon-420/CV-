import tensorflow as tf # TensorFlow 라이브러리 불러오기
from tensorflow.keras.datasets import cifar10 # CIFAR-10 데이터셋 불러오기
from tensorflow.keras.models import Sequential #레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #CNN에 필요한 레이더들 불러오기
from tensorflow.keras.preprocessing import image # 외부 이미지를 불러오기위한 라이브러리
import matplotlib.pyplot as plt #이미지 시각화 라이브러리
import numpy as np #배열 계산 및 데이터 처리용 라이브러리


# CIFAR-10 데이터셋의 클래스 숫자를 이름으로 정의
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# 데이터 전처리
# 픽셀 값을 0~255 범위에서 0~1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0


# CNN 모델 구성
model = Sequential([
    # 첫번째 합성곱층
    # 32개의 필터 이미지 특징 추출
    #입력 이미지 크기: 32x32, 채널 수: 3 (RGB)
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    # 첫 번째 풀링층
    # 이미지의 크기를 줄이고 중요한 정보만 유지
    MaxPooling2D((2, 2)),

    # 두번째 합성곱층
    # 더 복잡한 특징 추출
    Conv2D(64, (3, 3), activation='relu'),

    # 두 번째 풀링층
    # 크기 축소 및 연산량 감소
    MaxPooling2D((2, 2)),

    #2차원 특징맵을 1차원으로 변환(Dense 레이어 입력으로 사용하기 위해)
    Flatten(),
    # 완전 연결층
    Dense(64, activation='relu'),
    #출력층
    #10개의 클래스에 대한 확률출력
    Dense(10, activation='softmax')
])


# 모델 컴파일
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


# 모델 학습
history = model.fit(x_train, y_train,
    epochs=5, validation_data=(x_test, y_test))


# 테스트 데이터로 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print("테스트 정확도:", test_acc)


# dog.jpg 이미지 불러오기
# CIFAR-10 입력 크기에 맞게 32x32로 조정
img = image.load_img('dog.jpg', target_size=(32, 32))

# 이미지를 numpy 배열로 변환
img_array = image.img_to_array(img)

# 정규화
img_array = img_array / 255.0

# 배치 차원 추가 -> (1, 32, 32, 3)
img_input = np.expand_dims(img_array, axis=0)


# 8. dog.jpg 예측 수행
prediction = model.predict(img_input)

# 가장 확률이 높은 클래스 인덱스
predicted_label = np.argmax(prediction)

# 예측 확률
confidence = np.max(prediction)

#각 클래스에 대한 예측 확률 출력
for i, prob in enumerate(prediction[0]):
    print(f"{class_names[i]}: {prob*100:.2f}%")

# 결과 출력
plt.imshow(img)#이미지 표시
plt.title(f"predicted: {class_names[predicted_label]}")#예측 결과출력
plt.axis('off') #영역 표시 제거
plt.show() #이미지 시각화
