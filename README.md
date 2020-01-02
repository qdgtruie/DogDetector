# DogDetector
Small 3D printed prototype leveraging machine learning to build a treat dispenser for my dog 

### Required configuration ###
* raspberry pi 3 (raspberry pi 4 works faster but could overheat) 
* python 3

### Technologies ####
* Language programming: python 3
* Keras with a TensorFlow backend
* a custom keras model has been design for the task (see ```ModelBuilder.py```)

### Creating dataset  from your CLI ###
run ```sh helpers/Download_Images.py``` from CLI 

### Running learning from your CLI ###
run ```sh HookieLearner.py``` from CLI 

### Running detector from your CLI ###
run ```sh HookieDetector.py``` from CLI 
possible parameter are :
* ```-f``` or ```--faces``` to activate face recognition
* ```-g``` or ```--grayscale``` to enable grayscale image processing (improve result in some specific light conditions)
* ```-r``` or ```--revert``` to revert the image (in case you poorly setup the cam as I initially did)

### Overall approach  ###

1. Create a ```helpers/urls.txt``` based on google image.
2. Train the model (adjust ```MetaParams.py``` to your need)
3. Run the detector


### Possible improvements ###
0.TBC
