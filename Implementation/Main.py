import os

from Implementation.Model_Conv3d import model_conv3D
from Implementation.VideoReader import VideoReader


# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "VideoDataset")
model_path = os.path.join(abs_path, "Models")


if __name__ == "__main__":
    num_output_classes = 11
    train_reader = VideoReader(os.path.join(data_path,'train_map.csv'), num_output_classes, True)
    test_reader = VideoReader(os.path.join(data_path,'train_map.csv'), num_output_classes, True)
    print(train_reader)
    print(test_reader)
    model_conv3D()