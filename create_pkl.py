import os
import numpy as np
from PIL import Image
import pickle

# 경로 설정
custom_data_dir = "./original_Data"  # 직접 만든 PNG 이미지가 있는 폴더 경로
save_file ="original_Data.pkl"  # 저장할 pkl 파일 경로

def load_custom_dataset(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            image = Image.open(file_path)
            image = image.convert("L")  # 흑백 이미지로 변환
            image = image.resize((28, 28))  # 크기 조정
            image = np.array(image)
            image = image.reshape((1, 28, 28))  # 이미지 형태 변경
            image = image.astype(np.float32)
            images.append(image)

            label = int(filename.replace("-", "_").split("_")[0])
            
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def create_custom_mnist_pkl():
    images, labels = load_custom_dataset(custom_data_dir)
    # 데이터 분할
    split_ratio = 0.8  # 학습 데이터의 비율 설정
    split_index = int(len(images) * split_ratio)

    X_train = images[:split_index]
    Y_train = labels[:split_index]
    X_test = images[split_index:]
    Y_test = labels[split_index:]
    
    dataset = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test
    }

    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

create_custom_mnist_pkl()