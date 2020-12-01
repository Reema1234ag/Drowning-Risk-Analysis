# Drowning Risk Analysis
Drowning is one of the leading causes of unintentional death. It has become a serious issue in the past few years and has now become a major public health problem worldwide. Effective drowning detection methods are essential for the safety of the swimmers. Millions of people die every year due to this danger posing threat. Here, we are making an effort to develop an intelligent detection system using concepts of image processing, motion sensing, and machine learning algorithms to train our drowning detection model and provide an efficient and stable monitoring system and contribute to saving few lives.

The approach used here will be to alert the lifeguards with an alarm in case of drowning and decrease their reaction time. The developed model is made to ensure to detect drowning effectively and report at the earliest stage. For this a real-time underwater monitoring system is developed which can efficiently keep an eye on all the activities of the swimmer.

We started using the best object detection framework YOLO (You only look once).
The approach to this project is to first detect a person, drawing a blue rectangle around them. The program will now store the centre of that rectangle (the person's) position, and compare it to the person's position is more or less the same or they are falling, and this continues for 10 seconds, the blue rectangle will now turn red and the word 'DROWNING' will appear on top of it.
If the person´s centre is above water, the word ´DROWNING´ will be removed as they are just standing in the pool.

Various other algorithms can be used to detect the drowning with more appropriate accuracy rather than waiting for 10 seconds. Algorithms like LSTM, R-CNN along with YOLO to make it better as other than detection with the center of the bounding rectangle we can use height to width ratio to find the detection as the person who drowns appear to have more frequent changes in height to width ratio.
Various IoT techniques can be added to this program to make it a big lifeguard software which will not only detect but will give an on-time alert.
And Various privacy concerned factors can be added to it as well, as it will only detect not record.

This software can be used with Pi Camera, which can then be placed underwater to detect drowning.

Weights and cfg (or configuration) files can be downloaded from https://pjreddie.com/darknet/yolo. 
Names file can be downloaded from https://github.com/pjreddie/darknet/blob/master/data/coco.names.
