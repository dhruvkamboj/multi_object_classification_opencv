Project Title: Multi-Class Object Detection
Description:
This project uses the YOLO (You Only Look Once) model for multi-class object detection on a video. The goal is to detect objects in a video stream and annotate them with bounding boxes, labels, and counts of each detected object.

Requirements:
Python 3.x

Libraries:
opencv-python
ultralytics
collections

You can install the necessary libraries by running the following command:
pip install opencv-python ultralytics

Files:
detect.py: The Python script that uses YOLO for object detection in the given video.
people.mp4: Sample video used for detection.
yolov5su.pt: Pre-trained YOLOv5 model (can be replaced with any YOLOv5 model).

Steps to Run:

1.Clone or download the project files: Ensure you have the following files in your project directory:
your_code.py
people.mp4
yolov5su.pt

2.Run the Code:
In your terminal or command prompt, navigate to the directory where the files are stored and run the script:
python your_code.py

3.View Output:
The script will open a window showing the video with object detection annotations. You will see the bounding boxes and class labels for each detected object. Press q to quit the window.

4.Expected Output:
The program will display the video with:
Bounding boxes around detected objects.
Object class labels above each bounding box.
A count of each detected object class on the left side of the frame.

Datasets:
The COCO dataset is used for the class labels. The model uses the pre-trained yolov5su.pt, which is designed to work with the COCO class set.
link: https://cocodataset.org/#download
COCO Class List:
The model can detect a wide range of classes like 'person', 'car', 'cat', 'dog', etc., based on the COCO class definitions.

License:
This project is open-source and can be freely used for educational purposes.