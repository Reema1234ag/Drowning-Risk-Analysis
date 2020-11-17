from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint

from PIL import Image
import imageio

import cntk as C
from cntk.logging import *
from cntk.debugging import set_computation_network_trace_level

def conv3d_ucf11(train_reader, test_reader, max_epochs=30):
    # Replace 0 with 1 to get detailed log.
    # set_computation_network_trace_level(0)

    # These values must match for both train and test reader.
    image_height = train_reader.height
    image_width = train_reader.width
    num_channels = train_reader.channel_count
    sequence_length = train_reader.sequence_length
    num_output_classes = train_reader.label_count

    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, sequence_length, image_height, image_width), np.float32)
    label_var = C.input_variable(num_output_classes, np.float32)

    # Instantiate simple 3D Convolution network inspired by VGG network
    # and http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
    with C.default_options(activation=C.relu):
        z = C.layers.Sequential([
            C.layers.Convolution3D((3, 3, 3), 64, pad=True),
            C.layers.MaxPooling((1, 2, 2), (1, 2, 2)),
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution3D((3, 3, 3), [96, 128, 128][i], pad=True),
                C.layers.Convolution3D((3, 3, 3), [96, 128, 128][i], pad=True),
                C.layers.MaxPooling((2, 2, 2), (2, 2, 2))
            ]),
            C.layers.For(range(2), lambda: [
                C.layers.Dense(1024),
                C.layers.Dropout(0.5)
            ]),
            C.layers.Dense(num_output_classes, activation=None)
        ])(input_var)

    # loss and classification error.
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    # training config
    train_epoch_size = train_reader.size()
    train_minibatch_size = 2

    # Set learning parameters
    lr_per_sample = [0.01] * 10 + [0.001] * 10 + [0.0001]
    lr_schedule = C.learning_rate_schedule(lr_per_sample, epoch_size=train_epoch_size, unit=C.UnitType.sample)
    momentum_time_constant = 4096
    mm_schedule = C.momentum_as_time_constant_schedule([momentum_time_constant])

    # Instantiate the trainer object to drive the model training
    learner = C.momentum_sgd(z.parameters, lr_schedule, mm_schedule, True)
    ProgressPrinter = print_progress(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), learner, ProgressPrinter)

    # log_number_of_parameters(z)
    print()

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):  # loop over epochs
        train_reader.reset()

        while train_reader.has_more():
            videos, labels, current_minibatch = train_reader.next_minibatch(train_minibatch_size)
            trainer.train_minibatch({input_var: videos, label_var: labels})

        trainer.summarize_training_progress()

    # Test data for trained model
    epoch_size = test_reader.size()
    test_minibatch_size = 2

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    minibatch_index = 0

    test_reader.reset()
    while test_reader.has_more():
        videos, labels, current_minibatch = test_reader.next_minibatch(test_minibatch_size)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch({input_var: videos, label_var: labels}) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index + 1,
                                                                        (metric_numer * 100.0) / metric_denom,
                                                                        metric_denom))
    print("")

    return metric_numer / metric_denom
