import pickle
import numpy as np

# 첫 번째 pkl 파일 로드
with open('mnist.pkl', 'rb') as file:
    data1 = pickle.load(file)

# 두 번째 pkl 파일 로드
with open('original_Data.pkl', 'rb') as file:
    data2 = pickle.load(file)

# 데이터 형상 조정
data1['train_img'] = data1['train_img'].reshape(data1['train_img'].shape[0], 1, 28, 28)
data1['test_img'] = data1['test_img'].reshape(data1['test_img'].shape[0], 1, 28, 28)

# 데이터 합치기
combined_data = {}
combined_data['X_train'] = np.concatenate((data1['train_img'], data2['X_train']), axis=0)
combined_data['Y_train'] = np.concatenate((data1['train_label'], data2['Y_train']), axis=0)
combined_data['X_test'] = np.concatenate((data1['test_img'], data2['X_test']), axis=0)
combined_data['Y_test'] = np.concatenate((data1['test_label'], data2['Y_test']), axis=0)

# 합쳐진 데이터를 pkl 파일로 저장
with open('original_mnist.pkl', 'wb') as file:
    pickle.dump(combined_data, file)