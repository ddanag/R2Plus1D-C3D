import os
import random
import shutil
import zipfile

import rarfile
from distutils.dir_util import copy_tree

if not os.path.exists('/work/rclgroup/temp'):
    os.mkdir('/work/rclgroup/temp')

# preprocess ucf101 files
if not os.path.exists('/work/rclgroup/ucf101'):
    os.mkdir('/work/rclgroup/ucf101')
if not os.path.exists('/work/rclgroup/temp/ucf101'):
    os.mkdir('/work/rclgroup/temp/ucf101')

#ucf101_splits = zipfile.ZipFile('/work/rclgroup/UCF101TrainTestSplits-RecognitionTask.zip')
#ucf101_splits.extractall('/work/rclgroup/temp/ucf101')
#ucf101_splits.close()

#copy_tree('/work/rclgroup/UCF101TrainTestSplits-RecognitionTask', '/work/rclgroup/temp/ucf101')

if not os.path.exists('/work/rclgroup/ucf101_labels.txt'):
    with open('/work/rclgroup/ucf101_labels.txt', 'w') as f:
        for line in open('/work/rclgroup/temp/ucf101/ucfTrainTestlist/classInd.txt', 'r'):
        #for line in open('/work/rclgroup/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt', 'r'):
            f.write(line.split(' ')[1])
#import pdb; pdb.set_trace()
train_video_files, test_video_files = [], []
for line in open('/work/rclgroup/temp/ucf101/ucfTrainTestlist/trainlist01.txt', 'r'):
    train_video_files.append(line.split(' ')[0])

for line in open('/work/rclgroup/temp/ucf101/ucfTrainTestlist/testlist01.txt', 'r'):
    test_video_files.append(line.replace('\n', ''))

val_video_files = random.sample(test_video_files, int(len(test_video_files) * 0.2))

#ucf101_videos = rarfile.RarFile('/work/rclgroup/UCF101.rar')
#ucf101_videos.extractall('/work/rclgroup/temp/ucf101')
#ucf101_videos.close()

#copy_tree('/work/rclgroup/UCF-101', '/work/rclgroup/temp/ucf101')

if not os.path.exists('/work/rclgroup/ucf101/train'):
    os.mkdir('/work/rclgroup/ucf101/train')
for video in train_video_files:
    if not os.path.exists('/work/rclgroup/ucf101/train/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/ucf101/train/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/ucf101/UCF-101/{}'.format(video), '/work/rclgroup/ucf101/train/{}'.format(video))

if not os.path.exists('/work/rclgroup/ucf101/val'):
    os.mkdir('/work/rclgroup/ucf101/val')
for video in val_video_files:
    if not os.path.exists('/work/rclgroup/ucf101/val/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/ucf101/val/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/ucf101/UCF-101/{}'.format(video), '/work/rclgroup/ucf101/val/{}'.format(video))

if not os.path.exists('/work/rclgroup/ucf101/test'):
    os.mkdir('/work/rclgroup/ucf101/test')
for video in test_video_files:
    if not os.path.exists('/work/rclgroup/ucf101/test/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/ucf101/test/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/ucf101/UCF-101/{}'.format(video), '/work/rclgroup/ucf101/test/{}'.format(video))

"""
# preprocess hmdb51 files
if not os.path.exists('/work/rclgroup/hmdb51'):
    os.mkdir('/work/rclgroup/hmdb51')
if not os.path.exists('/work/rclgroup/temp/hmdb51'):
    os.mkdir('/work/rclgroup/temp/hmdb51')

hmdb51_splits = rarfile.RarFile('/work/rclgroup/test_train_splits.rar')
hmdb51_splits.extractall('/work/rclgroup/temp/hmdb51')
hmdb51_splits.close()

labels = []
for file in sorted(os.listdir('/work/rclgroup/temp/hmdb51/testTrainMulti_7030_splits')):
    labels.append(file.split('_test_split')[0])
labels = sorted(set(labels))

if not os.path.exists('/work/rclgroup/hmdb51_labels.txt'):
    with open('/work/rclgroup/hmdb51_labels.txt', 'w') as f:
        for current_label in labels:
            f.write(current_label + '\n')

train_video_files, val_video_files, test_video_files = [], [], []
for file in sorted(os.listdir('/work/rclgroup/temp/hmdb51/testTrainMulti_7030_splits')):
    if file.endswith('test_split1.txt'):
        for line in open('/work/rclgroup/temp/hmdb51/testTrainMulti_7030_splits/{}'.format(file), 'r'):
            if line.split(' ')[1].replace('\n', '') == '1':
                train_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '2':
                test_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '0':
                val_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])

hmdb51_videos = rarfile.RarFile('/work/rclgroup/hmdb51_org.rar')
hmdb51_videos.extractall('/work/rclgroup/temp/hmdb51')
hmdb51_videos.close()
for file in sorted(os.listdir('/work/rclgroup/temp/hmdb51/')):
    if file.endswith('.rar'):
        rar_file = rarfile.RarFile('/work/rclgroup/temp/hmdb51/{}'.format(file))
        rar_file.extractall('/work/rclgroup/temp/hmdb51')
        rar_file.close()

if not os.path.exists('/work/rclgroup/hmdb51/train'):
    os.mkdir('/work/rclgroup/hmdb51/train')
for video in train_video_files:
    if not os.path.exists('/work/rclgroup/hmdb51/train/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/hmdb51/train/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/hmdb51/{}'.format(video), '/work/rclgroup/hmdb51/train/{}'.format(video))

if not os.path.exists('/work/rclgroup/hmdb51/val'):
    os.mkdir('/work/rclgroup/hmdb51/val')
for video in val_video_files:
    if not os.path.exists('/work/rclgroup/hmdb51/val/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/hmdb51/val/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/hmdb51/{}'.format(video), '/work/rclgroup/hmdb51/val/{}'.format(video))

if not os.path.exists('/work/rclgroup/hmdb51/test'):
    os.mkdir('/work/rclgroup/hmdb51/test')
for video in test_video_files:
    if not os.path.exists('/work/rclgroup/hmdb51/test/{}'.format(video.split('/')[0])):
        os.mkdir('/work/rclgroup/hmdb51/test/{}'.format(video.split('/')[0]))
    shutil.copy('/work/rclgroup/temp/hmdb51/{}'.format(video), '/work/rclgroup/hmdb51/test/{}'.format(video))
"""
# remove the temp dir to make the data dir more clear
shutil.rmtree('/work/rclgroup/temp')
