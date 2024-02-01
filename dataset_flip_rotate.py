import os
'''
pip install cv2 오류나면 opencv-python 설치
pip install opencv-python
'''
import cv2
import numpy 
for i in range(0,10,1):
    for j in range(0,10,1):
        image_path = "./img4/"+str(i)+"_"+str(j)+'_original.png'
        image = cv2.imread(image_path)
        print(image)
        # 이미지 회전 (45도 반시계 방향으로 회전)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 이미지 뒤집기 (수평으로 뒤집기)
        flipped_image = cv2.flip(image, 1)

        # 회전된 이미지 저장

        # 이미지 저장
        cv2.imwrite("./img4/"+str(i)+"_"+str(j)+'_rotated.png', rotated_image)

        # 뒤집힌 이미지 저장

        cv2.imwrite('./img4/'+str(i)+'_'+str(j)+'_flipped.png', flipped_image)