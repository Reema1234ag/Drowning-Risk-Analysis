import csv
import os
import sys
from random import randint
from tkinter import Image

import imageio
import numpy as np


class VideoReader(object):
    '''
    A simple VideoReader:
    It iterates through each video and select 16 frames as
    stacked numpy arrays.
    Similar to http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
    '''

    def __init__(self, map_file, label_count, is_training, limit_epoch_size=sys.maxsize):
        '''
        Load video file paths and their corresponding labels.
        '''
        self.map_file = map_file
        self.label_count = label_count
        self.width = 112
        self.height = 112
        self.sequence_length = 16
        self.channel_count = 3
        self.is_training = is_training
        self.video_files = []
        self.targets = []
        self.batch_start = 0

        map_file_dir = os.path.dirname(map_file)

        with open(map_file) as csv_file:
            data = csv.reader(csv_file)
            for row in data:
                self.video_files.append(os.path.join(map_file_dir, row[0]))
                target = [0.0] * self.label_count
                target[int(row[1])] = 1.0
                self.targets.append(target)

        self.indices = np.arange(len(self.video_files))
        if self.is_training:
            np.random.shuffle(self.indices)
        self.epoch_size = min(len(self.video_files), limit_epoch_size)

    def size(self):
        return self.epoch_size

    def has_more(self):
        if self.batch_start < self.size():
            return True
        return False

    def reset(self):
        if self.is_training:
            np.random.shuffle(self.indices)
        self.batch_start = 0

    def next_minibatch(self, batch_size):
        '''
        Return a mini batch of sequence frames and their corresponding ground truth.
        '''
        batch_end = min(self.batch_start + batch_size, self.size())
        current_batch_size = batch_end - self.batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')

        inputs = np.empty(shape=(current_batch_size, self.channel_count, self.sequence_length, self.height, self.width),
                          dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.label_count), dtype=np.float32)
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
            targets[idx - self.batch_start, :] = self.targets[index]

        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size

    def _select_features(self, video_file):
        '''
        Select a sequence of frames from video_file and return them as
        a Tensor.
        '''
        video_reader = imageio.get_reader(video_file, 'ffmpeg')
        num_frames = len(video_reader)
        if self.sequence_length > num_frames:
            raise ValueError(
                'Sequence length {} is larger then the total number of frames {} in {}.'.format(self.sequence_length,
                                                                                                num_frames, video_file))

        # select which sequence frames to use.
        step = 1
        expanded_sequence = self.sequence_length
        if num_frames > 2 * self.sequence_length:
            step = 2
            expanded_sequence = 2 * self.sequence_length

        seq_start = int(num_frames / 2) - int(expanded_sequence / 2)
        if self.is_training:
            seq_start = randint(0, num_frames - expanded_sequence)

        frame_range = [seq_start + step * i for i in range(self.sequence_length)]
        video_frames = []
        for frame_index in frame_range:
            video_frames.append(self._read_frame(video_reader.get_data(frame_index)))

        return np.stack(video_frames, axis=1)

    def _read_frame(self, data):
        '''
        Based on http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
        We resize the image to 128x171 first, then selecting a 112x112
        crop.
        '''
        if (self.width >= 171) or (self.height >= 128):
            raise ValueError("Target width need to be less than 171 and target height need to be less than 128.")

        image = Image.fromarray(data)
        image.thumbnail((171, 128), Image.ANTIALIAS)

        center_w = image.size[0] / 2
        center_h = image.size[1] / 2

        image = image.crop((center_w - self.width / 2,
                            center_h - self.height / 2,
                            center_w + self.width / 2,
                            center_h + self.height / 2))

        norm_image = np.array(image, dtype=np.float32)
        norm_image -= 127.5
        norm_image /= 127.5

        # (channel, height, width)
        return np.ascontiguousarray(np.transpose(norm_image, (2, 0, 1)))