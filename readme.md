# Autonomous robot using deep learning

![Screenshot](https://i.imgur.com/IL0LoMd)

### This is an academic project that consists of building an autonomous robot capable of recongnizing and acting according to each road sign (currently it is trained to detect the stop and turn right signs)

## Required hardware
 
 * Raspberry pi 3B+
 * Pi Camera
 * 4-Wheel Robot Chassis Kit
 * wires

### This projects uses two different detection techniques

## Deep learning using SSD pretrained model (ssd mobilenet v2) [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
this model is very accurate but heavy on the raspberry pi so i retrained it on google colab and converted it to tensorlite
![Screenshot](https://i.imgur.com/VtOdDOh)


## Haar cascades
![Screenshot](https://i.imgur.com/B0PUeNW)