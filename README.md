
# Version of  [Emojinator](https://github.com/akshaybahadur21/Emojinator) trained for lesser classes
## With some bug fixes

Ref:
Original contributors:
##### 1) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 2) [Raghav Patnecha](https://github.com/raghavpatnecha)

This code helps you to recognize and classify different emojis. As of now, we are only supporting hand emojis.

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.


### Description
This uses Emojis code with updating few bugs and training with lesser classes. 
Emojis are ideograms and smileys used in electronic messages and web pages. Emoji exist in various genres, including facial expressions, common objects, places and types of weather, and animals. They are much like emoticons, but emoji are actual pictures instead of typographics.

### Functionalities
1) Filters to detect hand.
2) CNN for training the model.


### Python  Implementation

1) Network Used- Convolutional Neural Network

If you face any problem, kindly raise an issue

### Procedure

1) Install virtual environment with the following command

pip install virtualenv

2) Create a virtualenvironment

virtualenv -p python3 venv3

3) Activate the virtual environment

source venv3/bin/activate

4) make sure you are inside the project location, that is inside the folder Gesture-detection
cd Gesture-detection

5) Now install the dependencies from requirements.txt
Try,
pip install -r requirements.txt

If you face issues installing, consult us

6) Now open a python terminal 

python

7) Execute the following commands in python terminal
python
>>> import Video_Handler

Here you record the second gesture
>>> Video_Handler.save_gestures(0)

Here you record the first gesture
>>> Video_Handler.save_gestures(1)

Now you create a csv file corresponding to the gestures
>>> Video_Handler.createCSV_from_gestures()
After execution of above line, a file "train_foo.csv" should be created

Now train your model
>>> Video_Handler.train(2)
Because you trained with two gestures, the parameter passed is 2

Now see your model in action
>>> Video_Handler.start_gesture_recognition()











