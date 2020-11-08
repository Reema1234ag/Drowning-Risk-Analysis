from __future__ import print_function

from __future__ import print_function

import csv
import sys
import numpy as np
import imageio

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


from zipfile import ZipFile
import os


def download_and_extract(src):
    print ('Downloading ' + src)
    zip_file, h = urlretrieve(src, './delete.me')
    print ('Done downloading, start extracting.')
    try:
        with ZipFile(zip_file, 'r') as zfile:
            zfile.extractall('.')
            print ('Done extracting.')
    finally:
        os.remove(zip_file)

def load_groups(input_folder):
    '''
    Load the list of sub-folders into a python list with their
    corresponding label.
    '''
    groups         = []
    label_folders  = os.listdir(input_folder)
    index          = 0
    for label_folder in sorted(label_folders):
        label_folder_path = os.path.join(input_folder, label_folder)
        if os.path.isdir(label_folder_path):
            group_folders = os.listdir(label_folder_path)
            for group_folder in group_folders:
                if group_folder != 'Annotation':
                    groups.append([os.path.join(label_folder_path, group_folder), index])
            index += 1

    return groups

def split_data(groups, file_ext):
    '''
    Split the data at random for train, eval and test set.
    '''
    group_count = len(groups)
    indices = np.arange(group_count)

    np.random.seed(0) # Make it deterministic.
    np.random.shuffle(indices)

    # 80% training and 20% test.
    train_count = int(0.8 * group_count)
    test_count  = group_count - train_count

    train = []
    test  = []

    for i in range(train_count):
        group = groups[indices[i]]
        video_files = os.listdir(group[0])
        for video_file in video_files:
            video_file_path = os.path.join(group[0], video_file)
            if os.path.isfile(video_file_path):
                video_file_path = os.path.abspath(video_file_path)
                ext = os.path.splitext(video_file_path)[1]
                if (ext == file_ext):
                    # make sure we have enough frames and the file isn't corrupt
                    video_reader = imageio.get_reader(video_file_path, 'ffmpeg')
                    if len(video_reader) >= 16:
                        train.append([video_file_path, group[1]])

    for i in range(train_count, train_count + test_count):
        group = groups[indices[i]]
        video_files = os.listdir(group[0])
        for video_file in video_files:
            video_file_path = os.path.join(group[0], video_file)
            if os.path.isfile(video_file_path):
                video_file_path = os.path.abspath(video_file_path)
                ext = os.path.splitext(video_file_path)[1]
                if (ext == file_ext):
                    # make sure we have enough frames and the file isn't corrupt
                    video_reader = imageio.get_reader(video_file_path, 'ffmpeg')
                    if len(video_reader) >= 16:
                        test.append([video_file_path, group[1]])

    return train, test

def write_to_csv(items, file_path):
    '''
    Write file path and its target pair in a CSV file format.
    '''
    if sys.version_info[0] < 3:
        with open(file_path, 'wb') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for item in items:
                writer.writerow(item)
    else:
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for item in items:
                writer.writerow(item)

def generate_and_save_labels():
    groups = load_groups('./action_youtube_naudio')
    train, test = split_data(groups, '.avi')

    write_to_csv(train, os.path.join('.', 'train_map.csv'))
    write_to_csv(test, os.path.join('.', 'test_map.csv'))

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    download_and_extract('https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip')
    print ('Writing train and test CSV file...')
    generate_and_save_labels()
    print ('Done.')