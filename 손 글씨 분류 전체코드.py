import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings 

#경고 무시
warnings.filterwarnings(action='ignore')

train_file_path = r"/content/drive/MyDrive/Colab Notebooks/train/train_data.csv"
test_file_path = r"/content/drive/MyDrive/Colab Notebooks/test/test_data.csv"
sample_submission_file_path = r"/content/drive/MyDrive/Colab Notebooks/sample_submission.csv"
train_dir = r"/content/drive/MyDrive/Colab Notebooks/train"
test_dir = r"/content/drive/MyDrive/Colab Notebooks/test"

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(sample_submission_file_path)

display(train, test)

from matplotlib import pyplot as plt
train['label'].plot(kind='hist', bins=20, title='label')
plt.gca().spines[['top', 'right',]].set_visible(False)

train['label'].value_counts()

file_name = train['file_name'][0]
file_path = f"{train_dir}/{file_name}"

try:
    sample_image = Image.open(file_path)
    sample_label = train['label'][0]
    plt.title('label: ' + str(sample_label))
    plt.imshow(sample_image, cmap='gray')
    plt.show()
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path}")

# 데이터셋 생성 함수

def load_image(image_file, label, img_size=(28, 28)):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)  # 흑백 이미지는 채널을 1로 설정
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # 부동 소수점 타입으로 변환 및 스케일링
    return image, label

# 데이터셋 로드
def create_dataset(file_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# 학습 및 검증 데이터셋 생성
train_image_paths = [os.path.join(train_dir, fname) for fname in train['file_name']]
train_labels = train['label'].values
validation_split = 0.2
num_train = int((1 - validation_split) * len(train_image_paths))

train_dataset = create_dataset(train_image_paths[:num_train], train_labels[:num_train])
validation_dataset = create_dataset(train_image_paths[num_train:], train_labels[num_train:])

# 테스트 데이터셋 생성
test_image_paths = [os.path.join(test_dir, fname) for fname in test['file_name']]


# 모델 설계
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer="rmsprop",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=15, validation_data=validation_dataset)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()

# 데이터 증강 설정
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

# 모델 설계 (데이터 증강 추가)
model_augmented = Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    data_augmentation,  # 데이터 증강 추가
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model_augmented.compile(optimizer="rmsprop",
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# 콜백 설정
callbacks = ModelCheckpoint(
    filepath='best_model_augmented.h5',
    save_best_only=True,
    monitor='val_loss')

history_augmented = model_augmented.fit(train_dataset, epochs=50, validation_data=validation_dataset, callbacks=callbacks)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_augmented.history['loss'], label='train_loss')
plt.plot(history_augmented.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs (Augmented Data)')

plt.subplot(1, 2, 2)
plt.plot(history_augmented.history['accuracy'], label='train_accuracy')
plt.plot(history_augmented.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs (Augmented Data)')


# 예측 수행 (데이터 증강)
predictions_augmented = model_augmented.predict(test_dataset)
predicted_labels_augmented = np.argmax(predictions_augmented, axis=1)

# test 데이터셋의 파일 이름 추출
file_paths = [os.path.basename(file_path) for file_path in test_image_paths]

# 제출 파일 생성 (데이터 증강)
submission_augmented = pd.DataFrame({
    'file_name': file_paths,
    'label': predicted_labels_augmented
})
submission_augmented.to_csv('submission_augmented.csv', index=False)
print("Submission file with augmented data created")