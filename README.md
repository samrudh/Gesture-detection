
# Gesture Detector - Automated feedback system based on Gesture



This code helps you to recognize and classify different emojis. As of now, we are only supporting hand emojis.



### Description
This project tries to understand user feedback from the gestures he/she shows with her hands. Feel free to train the model on your favorite hand gestures and see if it can detect the same later on when your friends make the same gestures.

### Functionalities
1) Filters to detect hand.
2) CNN for training the model.


### Python  Implementation

1) Network Used- Convolutional Neural Network



### Procedure

1) Install virtual environment with the following command

```sh
$ pip install virtualenv
```


2) Create a virtualenvironment

```sh
$ virtualenv -p python3 venv3
```


3) Activate the virtual environment
```sh
$ source venv3/bin/activate
```


4) make sure you are inside the project location, that is inside the folder Gesture-detection

```sh
$ cd Gesture-detection
```


5) Now install the dependencies from requirements.txt
Try,

```sh
$ pip install -r requirements.txt
```



If you face issues installing, consult us

6) Now open a python terminal 

```sh
$ python
```




7) Execute the following commands in python terminal

```sh
>>> import Video_Handler
```

Before going to the next step please ensure that the folder "gestures" is empty.

Here you record the first gesture

```sh
>>> Video_Handler.save_gestures(0)
```



Here you record the second gesture

```sh
>>> Video_Handler.save_gestures(1)
```


Now you create a csv file corresponding to the gestures

```sh
>>> Video_Handler.createCSV_from_gestures()
```


After execution of above line, a file "train_foo.csv" should be created

Now train your model

```sh
>>> Video_Handler.train(2)
```
Because you trained with two gestures, the parameter passed is 2

Now see your model in action

```sh
>>> Video_Handler.start_gesture_recognition()
```



Credits and refrences:
Original contributors:
##### 1) [Alexander Mordvintsev & Abid K. Revision 43532856](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)
##### 2) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 3) [Raghav Patnecha](https://github.com/raghavpatnecha)












