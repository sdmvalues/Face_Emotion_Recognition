# Face_Emotion_Recognition


Please download shape_predictor_68_face_landmarks.dat file from https://drive.google.com/file/d/1hyDn8eJ5yaTVkMgdKGmoFIn48zwdvIkg/view

# Dependencies
- Opencv
- Dlib
- face_recognition
- Keras

# Usage
- Create an 'images' folder for face_recognition and save jpg/png files under this folder. 
- models contain the pre-trained model for emotion classifier.
- emotion.py can to run to classify emotions of person's face.
- face-rec-emotion.py can recognise faces and classify emotion at a time.

## Run face_rec_emotion.py that would switch on the webcam and tag emotions in the existing setup

### Line no 311 of face_rec_emotion.py is:
    cap = cv2.VideoCapture(0) # Webcam source
    '0' is for computer webcam, and kindly change it to '1' of connected to external webcam
    
### line No. 100 and below under  datasets.py under utils got labellings for emotions. This can be changed
