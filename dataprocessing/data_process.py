import sys
import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import IPython.display as display
from tqdm import tqdm
import random
import sys
import glob

def data_to_tfrecord(imagepath, outputpath, mode):
    # 여러 개의 폴더에 있는 것 합치기
    images, label_list = [], []
    path = imagepath
    for folder in os.listdir(path):
        i = 0       
        landmark_df = pd.read_csv(glob.glob(path+folder+"/*.csv")[0])        
        landmark_columns = [point for point in landmark_df.columns if "-" in point]
        landmark_points = []
        for image in tqdm(os.listdir(path+folder), desc=folder+" 폴더 작업중"):     
            # if '+0' in folder:
            if '.csv' in image: #landmark point
                continue 
            if np.nan in landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))]:
                continue
            temp = landmark_df[landmark_df['frame'] == float(image.replace(".jpg", ""))]
            x1 = temp['right'] - temp['left']
            y1 = temp['bottom'] - temp['top']
            for col in landmark_columns:
                if 'x' in col:
                    landmark_points.append((float(temp[col]) - float(temp['left'])) / float(x1))
                elif 'y' in col:
                    landmark_points.append((float(temp[col]) - float(temp['top'])) / float(y1))            
            # print(landmark_points)
            images.append({'path': path+folder+"/"+image, 'class': int(folder.split('+')[1]), 'landmark_points': landmark_points})
            landmark_points = []
            i += 1    

    # write tfrecord
    save_path = outputpath
    i = 0
    random.shuffle(images)
    with tf.io.TFRecordWriter(save_path) as f:
        for image in tqdm(images, desc='이미지를 tfrecord에 저장중'):
            label = [0] * 7
            image_landmark = image['landmark_points']
            if mode == 'multi':
                label[image['class']] = 1
                image_class = label
            else:
                image_class = image['class']
            # print(image['landmark_points'])
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
                if mode == 'multi':
                    record = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file])),
                                'class': tf.train.Feature(int64_list=tf.train.Int64List(value=image_class)),
                                'landmark_points': tf.train.Feature(float_list=tf.train.FloatList(value=image_landmark))
                                # 'bb_box': tf.train.Feature(float_list=tf.train.FloatList(value=bb_box_list)),
                            }
                        )
                    )
                else:
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
    print([i['class'] for i in images])