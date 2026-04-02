import tensorflow as tf  # TensorFlow 라이브러리
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋
from tensorflow.keras.models import Sequential  # 모델 구조
from tensorflow.keras.layers import Dense  # Dense 레이어

# x_train: 학습용 이미지, y_train: 학습용 라벨
# x_test: 테스트 이미지, y_test: 테스트 라벨
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화 (0~255 → 0~1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 1차원 변환 (28x28 → 784)
# Dense 레이어는 1차원 입력만 받기 때문에 이미지 데이터를 1차원으로 변환해야 함
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#신경망 모델 생성
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), #입력데이터784개, 변환데이터128개
    Dense(64, activation='relu'), #변환데이터128개, 변환데이터64개로 변환
    Dense(10, activation='softmax') #변환데이터64개, 출력데이터10개 (0~9 숫자 분류)
    #각 클래스일 확률로 변환
])

#모델 컴파일 (가중치, 손실함수, 정확도 측정)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

#모델 학습 (입력데이터, 정답, 전체 데이터 5회 반복)
model.fit(x_train, y_train, epochs=5)
#모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
#최종 정확도 출력
print("테스트 정확도:", test_acc)