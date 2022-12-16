# Facial-Expression-Recognition
## 폴더 설명
* **dataprocessing 폴더**: 데이터 전처리 관련한 폴더, 딥러닝 모델 사용할때만 이용함
* **feature_change 폴더**: 점이나, 거리 쌍이 어떻게 변화하는지 시각화를 통해 확인해본 코드, Rule 기반 코드 포함되어 있음
* **modeling 폴더**: 이미지, 점, 거리쌍을 이용해 모델링을 진행한 폴더
    * **using_image 폴더**: 이미지만을 사용해 딥러닝 모델에 적용한 코드
    * **using_image_multi 폴더**: 이미지와 점을 사용해 딥러닝 모델에 적용한 코드
    * **using_table 폴더**: 점, 거리쌍을 이용해 머신러닝 모델에 적용한 코드

## 주요 파일 설명 

```
├─dataprocessing
│  │  change_tfrecords.ipynb
│  │  data_process.py
│  │  image_open.ipynb
│  │  make_tfrecords.ipynb
│  │  
│  └─__pycache__
│          data_process.cpython-38.pyc
│          data_process.cpython-39.pyc
│          read_data.cpython-38.pyc
│          
├─feature_change
│      dist_change.ipynb
│      emo_dist_change.ipynb
│      emo_dist_change_table.ipynb
│      half_dist_change.ipynb
│      point_change.ipynb
│      point_change_windowing.ipynb
│      
└─modeling
    │  importance_landmark.csv
    │  model.png
    │  
    │      
    ├─using_image
    │      cnnmodel.ipynb
    │      cnnmodel_testdata.ipynb
    │      moblienetV2.ipynb
    │      moblienetV2_landmark.ipynb
    │      moblienetV2_landmark_cam.ipynb
    │      moblienetV2_landmark_regress.ipynb
    │      
    ├─using_image_multi
    │      moblienetV2_landmark_multi.ipynb
    │      
    └─using_table
            6개표정.ipynb
            6개표정_함수화.ipynb
            6개표정_함수화_rule작성.ipynb
            dist_change.ipynb
            dist_change_regression.ipynb
            point_change.ipynb
```