import sys
import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import IPython.display as display
from tqdm import tqdm
import random
import sys

def data_to_tfrecord(imagepath, outputpath):
    # 여러 개의 폴더에 있는 것 합치기
    images, label_list = [], []
    path = imagepath
    for folder in os.listdir(path):
        i = 0       
        landmark_df = pd.read_csv(path+folder+"/"+folder.split("+")[0]+".csv")
        landmark_columns = [point for point in landmark_df.columns if "-" in point] 
        for image in tqdm(os.listdir(path+folder), desc=folder+" 폴더 작업중"):     
            if '+0' in folder:
                if '.csv' in image: #landmark point
                    continue 
                if np.nan in landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))]:
                    continue
                images.append({'path': path+folder+"/"+image, 'class': 0, 'landmark_points': landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))][landmark_columns].to_numpy()[0]})
                label_list.append(0)
            elif '+5' in folder:
                if '.csv' in image: #landmark point                    
                    continue
                if np.nan in landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))]:
                    continue
                images.append({'path': path+folder+"/"+image, 'class': 0.5, 'landmark_points': landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))][landmark_columns].to_numpy()[0]})
                label_list.append(0.5)
            elif '+1' in folder:
                if '.csv' in image: #landmark point
                    # landmark_df = pd.read_csv(image)
                    # landmark_columns = [point for point in landmark_df.columns if "-" in point]
                    continue
                if np.nan in landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))]:
                    continue
                images.append({'path': path+folder+"/"+image, 'class': 1, 'landmark_points': landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))][landmark_columns].to_numpy()[0]})
                label_list.append(1)
            i += 1
    # print('num_classes: ', len(set(label_list)))
    # encoder = class_one_hot_encoder(label_list)

    # write tfrecord
    save_path = outputpath
    i = 0
    random.shuffle(images)
    with tf.io.TFRecordWriter(save_path) as f:
        for image in tqdm(images, desc='이미지를 tfrecord에 저장중'):

            image_class = image['class']
            # print(image['landmark_points'])
            image_landmark = image['landmark_points']
            # image_class = float(image_class)

            try:
                # image
                image_file = tf.io.decode_image(tf.io.read_file(image['path'])).numpy()
                image_file = tf.image.resize(image_file, size=[224, 224])
                image_file = tf.io.encode_jpeg(tf.cast(image_file, dtype=tf.uint8)).numpy()
                # image_width = int(image['@width'])
                # image_height = int(image['@height'])

                # image_box = image['box']
                # top = int(image_box['@top'])
                # left = int(image_box['@left'])
                # width = int(image_box['@width'])
                # height = int(image_box['@height'])
                # bb_box_list = [top, left, width, height]
                # bb_box_list = point_adjust(bb_box_list, image_width, image_height)

                # # 'head_top', 'lear_base', 'lear_tip', 'leye, 'nose', 'rear_base', 'rear_tip', 'reye'
                # # x,y 순서
                # landmark_list = []
                # for point in image_box['part']:
                #     x = int(point['@x'])
                #     y = int(point['@y'])
                #     landmark_list.extend([x, y])
                # landmark_list = point_adjust(landmark_list, image_width, image_height)

                # record
                record = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file])),
                            'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_class])),
                            'landmark_points': tf.train.Feature(float_list=tf.train.FloatList(value=image_landmark))
                            # 'bb_box': tf.train.Feature(float_list=tf.train.FloatList(value=bb_box_list)),
                        }
                    )
                )
                f.write(record.SerializeToString())
                i += 1
            except Exception as e:
                print(e, i)
                continue

    print('이미지 갯수: ', i)
    print(label_list)