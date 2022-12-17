# Facial-Expression-Recognition
## 폴더 설명
* **dataprocessing 폴더**: 데이터 전처리 관련한 폴더, 딥러닝 모델 사용할때만 이용함
* **feature_change 폴더**: 동일 좌표나, 거리쌍이 어떻게 변화하는지 시각화를 통해 확인해본 코드, Rule 기반 코드 포함되어 있음
* **modeling 폴더**: 이미지, 동일 좌표, 거리쌍을 이용해 모델링을 진행한 폴더
    - **using_image 폴더**: 이미지만을 사용해 딥러닝 모델에 적용한 코드
    - **using_image_multi 폴더**: 이미지와 좌표을 사용해 딥러닝 모델에 적용한 코드
    - **using_table 폴더**: 동일 좌표, 거리쌍을 이용해 머신러닝 모델에 적용한 코드

## 주요 파일 설명 

```
├─dataprocessing
│      change_tfrecords.ipynb
│      data_process.py
│      image_open.ipynb
│      make_tfrecords.ipynb
│          
├─feature_change
│      dist_change.ipynb: 거리 쌍의 변화 기반 시각화 코드 
│      emo_dist_change.ipynb: Rule 기반 접근법을 위해 사용한 코드 (시각화, rule json 작성)
│      emo_dist_change_table.ipynb
│      half_dist_change.ipynb: 편마비 확인을 위한 데이터 시각화 코드
│      point_change.ipynb: 동일 좌표 이동 거리 기반 시각화 코드
│      point_change_windowing.ipynb: 동일 좌표 이동 거리 기반 window기법 적용 시각화 코드
│      
└─modeling
    │  importance_landmark.csv
    │  model.png
    │      
    ├─using_image
    │      cnnmodel.ipynb: CNN으로 이미지만을 통해 분류 진행한 코드  
    │      cnnmodel_testdata.ipynb: CNN으로 공개된 이미지 데이터를 이용해 분류 진행한 코드
    │      moblienetV2.ipynb: moblienetV2로 이미지만을 통해 분류 진행한 코드 
    │      
    ├─using_image_multi
    │      moblienetV2_landmark_multi.ipynb: moblienetV2로 이미지와 좌표를 넣어서 학습 후 예측하는 코드
    │      moblienetV2_landmark_cam.ipynb: moblienetV2로 이미지와 좌표를 넣어서 학습 후 cam구조를 통해 시각화하는 코드
    │      moblienetV2_landmark_regress.ipynb
    │      
    └─using_table
            6개표정.ipynb
            6개표정_함수화.ipynb: 6개 표정에 대해서 함수화를 통한 점수 출력 시도해본 코드
            6개표정_함수화_rule작성.ipynb
            dist_change.ipynb: 거리 쌍의 변화 기반 머신러닝(RF, XGBoost) 모델 코드
            dist_change_regression.ipynb: 거리 쌍의 변화 기반 머신러닝(RF, XGBoost) 회귀 모델 코드
            point_change.ipynb: 동일 좌표 이동거리 기반 머신러닝(RF, XGBoost) 모델 코드
```

# 데이터 설명
```
📦data
 ┣ 📂image_analysis: Rule 기반 접근법에 사용한 데이터 (다양한 사람들의 4가지 감정 표정)
 ┣ 📂image_mobilenet: 딥러닝 접근법에 사용한 데이터 (7가지 표정)
 ┣ 📂images: 머신 러닝 접근법에 사용한 원본 데이터 (6가지 표정)
 ┣ 📂images2: 딥러닝 접근법에 사용한 원본 데이터
 ┣ 📜face_data.tfrecord: 딥러닝 접근법에 사용한 데이터 (이미지)
 ┣ 📜face_data2.tfrecord
 ┣ 📜face_data3.tfrecord
 ┣ 📜face_data4.tfrecord
 ┣ 📜face_data5.tfrecord
 ┣ 📜face_data6.tfrecord
 ┣ 📜face_data7.tfrecord
 ┣ 📜face_data_multi.tfrecord: 딥러닝 접근법에 사용한 데이터 (이미지, 좌표 포함)
 ┣ 📜face_data_regress.tfrecord
 ┗ 📜merge.tfrecord
```
