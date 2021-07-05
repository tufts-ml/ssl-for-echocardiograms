import numpy as np
import pandas as pd
import os
from tqdm import trange
import PIL.Image as Image
from easydict import EasyDict as edict

import argparse
import tensorflow as tf

print(tf.__version__)
parser = argparse.ArgumentParser()
parser.add_argument('--result_save_root_dir', default='../ML_DATA', help="where to save the processed tfrecord")
parser.add_argument('--dataset_name', default='TMED-18-18', help="TMED-18-18 or TMED-156-52")
parser.add_argument('--fold', default='fold0')
parser.add_argument('--raw_data_dir', default='../raw_data', help="where the raw png files are saved")
parser.add_argument('--suggested_split_file_dir', default='../raw_data/SplitImageLabelMapping/')


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def LoadImageFeature(file_path):
    im = Image.open(file_path)
    im = np.asarray(im)
    im = im[:,:, np.newaxis] #make it (64, 64, 1) 1:channel
    return im

def check_unlabeled_origin(image_name, raw_data_dir, dataset='TMED-156-52'):
    '''
    check if the unlabeled data comes from unlabeled/ or partially_labeled/ or labeled/
    '''
    print(image_name)
    assert dataset in ['TMED-156-52', 'TMED-18-18'], 'Currently only released two versions of the TMED dataset'
    if dataset == 'TMED-156-52':
        unlabeled_flag = False
        partiallylabeled_flag = False

        images_unlabeled = os.listdir(os.path.join(raw_data_dir, 'unlabeled'))
        images_partiallylabeled = os.listdir(os.path.join(raw_data_dir, 'partially_labeled'))
        if image_name in images_unlabeled:
#             print('{} in unlabeled/'.format(image_name))
            unlabeled_flag = True

        if image_name in images_partiallylabeled:
#             print('{} in partiallylabeled/'.format(image_name))
            partiallylabeled_flag = True

        assert unlabeled_flag != partiallylabeled_flag, 'the unlabeled image must be in one and only one of the unlabeled/ or partially_labeled/'

        if unlabeled_flag:
            return 'unlabeled'
        else:
            return 'partially_labeled'
    
    else: #'TMED-18-18'
        labeled_flag = False
        partiallylabeled_flag = False

        images_labeled = os.listdir(os.path.join(raw_data_dir, 'labeled'))
        images_partiallylabeled = os.listdir(os.path.join(raw_data_dir, 'partially_labeled'))
        if image_name in images_labeled:
#             print('{} in labeled/'.format(image_name))
            labeled_flag = True

        if image_name in images_partiallylabeled:
#             print('{} in partiallylabeled/'.format(image_name))
            partiallylabeled_flag = True

        assert labeled_flag != partiallylabeled_flag, 'the unlabeled image must be in one and only one of the unlabeled/ or partially_labeled/'

        if labeled_flag:
            return 'labeled'
        else:
            return 'partially_labeled'
    
    
def main(args_dict):
    
    result_save_root_dir = args_dict.result_save_root_dir
    dataset_name = args_dict.dataset_name
    fold = args_dict.fold
    raw_data_dir = args.raw_data_dir
    suggested_split_file_dir = args.suggested_split_file_dir
    
    result_save_dir = os.path.join(result_save_root_dir, dataset_name, fold)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    
    view_label_to_class_mapping = {'PLAX':0, 'PSAX AoV':1, 'Other':2}
    diagnosis_label_to_class_mapping = {'no_as':0, 'mild/moderate_as':1, 'severe_as':2}
    
    train_image_list = []
    train_diagnosis_labels_list = []
    train_view_labels_list = []

    val_image_list = []
    val_diagnosis_labels_list = []
    val_view_labels_list = []

    test_image_list = []
    test_diagnosis_labels_list = []
    test_view_labels_list = []

    unlabeled_image_list = []

    
    #read from the suggested split csv
    suggested_split_csv = pd.read_csv(os.path.join(suggested_split_file_dir, '{}_{}.csv'.format(dataset_name, fold)))
    
    for i in trange(suggested_split_csv.shape[0]):
        query_key = suggested_split_csv.iloc[i].query_key
        split = suggested_split_csv.iloc[i].split
    
        if split in ['train', 'val', 'test']:
            data_folder = 'labeled'
            diagnosis_label = suggested_split_csv.iloc[i].diagnosis_label
            diagnosis_label_class = diagnosis_label_to_class_mapping[diagnosis_label] 
            view_label = suggested_split_csv.iloc[i].view_label
            view_label_class = view_label_to_class_mapping[view_label]
        
        elif split == 'Unlabeled':
            data_folder = check_unlabeled_origin(query_key, raw_data_dir, dataset_name)
            
        
        im = LoadImageFeature(os.path.join(raw_data_dir, data_folder, query_key))
        #distribute to different trainlabeled, val, test, unlabeled
        if split == 'train':
            train_image_list.append(im)
            train_diagnosis_labels_list.append(diagnosis_label_class)
            train_view_labels_list.append(view_label_class)
        elif split == 'val':
            val_image_list.append(im)
            val_diagnosis_labels_list.append(diagnosis_label_class)
            val_view_labels_list.append(view_label_class)
        elif split == 'test':
            test_image_list.append(im)
            test_diagnosis_labels_list.append(diagnosis_label_class)
            test_view_labels_list.append(view_label_class)
        else:
            unlabeled_image_list.append(im)

    train_image_list = np.array(train_image_list)
    val_image_list = np.array(val_image_list)
    test_image_list = np.array(test_image_list)
    unlabeled_image_list = np.array(unlabeled_image_list)

    train_size = len(train_image_list)
    val_size = len(val_image_list)
    test_size = len(test_image_list)
    unlabeled_size = len(unlabeled_image_list)

    train_image_list = _encode_png(train_image_list)
    val_image_list = _encode_png(val_image_list)
    test_image_list = _encode_png(test_image_list)
    unlabeled_image_list = _encode_png(unlabeled_image_list)
    
    
    #Save into tfrecord
    
    #DIAGNOSIS
    #train tfrecord
    train_tfrecord_fullpath = os.path.join(result_save_dir, 'train_DIAGNOSIS.tfrecord')
    with tf.python_io.TFRecordWriter(train_tfrecord_fullpath) as writer:
        for i in trange(train_size, desc = 'WRITING train diagnosis tfrecords'):
            feat = dict(image = _bytes_feature(train_image_list[i]),
                        label=_int64_feature(train_diagnosis_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', train_tfrecord_fullpath)

    val_tfrecord_fullpath = os.path.join(result_save_dir, 'val_DIAGNOSIS.tfrecord')
    with tf.python_io.TFRecordWriter(val_tfrecord_fullpath) as writer:
        for i in trange(val_size, desc = 'WRITING val diagnosis tfrecords'):
            feat = dict(image = _bytes_feature(val_image_list[i]),
                        label=_int64_feature(val_diagnosis_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', val_tfrecord_fullpath)

    test_tfrecord_fullpath = os.path.join(result_save_dir, 'test_DIAGNOSIS.tfrecord')
    with tf.python_io.TFRecordWriter(test_tfrecord_fullpath) as writer:
        for i in trange(test_size, desc = 'WRITING test diagnosis tfrecords'):
            feat = dict(image = _bytes_feature(test_image_list[i]),
                        label=_int64_feature(test_diagnosis_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', test_tfrecord_fullpath)
    
    #VIEW
    #train tfrecord
    train_tfrecord_fullpath = os.path.join(result_save_dir, 'train_VIEW.tfrecord')
    with tf.python_io.TFRecordWriter(train_tfrecord_fullpath) as writer:
        for i in trange(train_size, desc = 'WRITING train view tfrecords'):
            feat = dict(image = _bytes_feature(train_image_list[i]),
                        label=_int64_feature(train_view_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', train_tfrecord_fullpath)

    val_tfrecord_fullpath = os.path.join(result_save_dir, 'val_VIEW.tfrecord')
    with tf.python_io.TFRecordWriter(val_tfrecord_fullpath) as writer:
        for i in trange(val_size, desc = 'WRITING val view tfrecords'):
            feat = dict(image = _bytes_feature(val_image_list[i]),
                        label=_int64_feature(val_view_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', val_tfrecord_fullpath)

    test_tfrecord_fullpath = os.path.join(result_save_dir, 'test_VIEW.tfrecord')
    with tf.python_io.TFRecordWriter(test_tfrecord_fullpath) as writer:
        for i in trange(test_size, desc = 'WRITING test view tfrecords'):
            feat = dict(image = _bytes_feature(test_image_list[i]),
                        label=_int64_feature(test_view_labels_list[i])
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', test_tfrecord_fullpath)
    
    #unlabeled tfrecrod
    unlabeled_tfrecord_fullpath = os.path.join(result_save_dir, 'unlabeled.tfrecord')
    with tf.python_io.TFRecordWriter(unlabeled_tfrecord_fullpath) as writer:
        for i in trange(unlabeled_size, desc = 'WRITING unlabeled tfrecords'):
            feat = dict(image = _bytes_feature(unlabeled_image_list[i]),
                        label=_int64_feature(-1)
            )

            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())

    print('FINISHED SAVING: ', unlabeled_tfrecord_fullpath)
    
    

if __name__=='__main__':
    
    args = parser.parse_args()
    result_save_root_dir = args.result_save_root_dir
    dataset_name = args.dataset_name #'TMED-18-18', 'TMED-156-52'
    fold = args.fold # fold0, fold1, fold2 etc
    raw_data_dir = args.raw_data_dir
    suggested_split_file_dir = args.suggested_split_file_dir
      
    input_args_dict = edict()
    input_args_dict.result_save_root_dir = result_save_root_dir
    input_args_dict.dataset_name = dataset_name
    input_args_dict.fold = fold
    input_args_dict.raw_data_dir = raw_data_dir
    input_args_dict.suggested_split_file_dir = suggested_split_file_dir
    
    main(input_args_dict)
    
    
    
    