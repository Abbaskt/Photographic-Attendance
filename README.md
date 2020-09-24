# Photographic Attendance

## Working

First run the gather_image.py to run collect 30 sets of photographs of individual students with different expressions.

After taking photographs of all students train a model using the train.py file which will create a model.yml file.

Then take a photo of a class consisting of all students present in the dataset and use that image to extract individual images of students using extract_image.py

Finally feed those image to the model to predict the name of the students using detect.py.

      
