import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# # Load a sample picture and learn how to recognize it.
# soum_image = face_recognition.load_image_file("images/soum.jpg")
# soum_face_encoding = face_recognition.face_encodings(soum_image)[0]
#
# a = "images/yusuf.png"
# Yusuf_image = face_recognition.load_image_file(a)
# Yusuf_face_encoding = face_recognition.face_encodings(Yusuf_image)[0]
#
# b = "images/sarit.jpg"
# sarit_image = face_recognition.load_image_file(b)
# sarit_face_encoding = face_recognition.face_encodings(sarit_image)[0]
#
# c = "images/Ben.jpg"
# Ben_image = face_recognition.load_image_file(c)
# Ben_face_encoding = face_recognition.face_encodings(Ben_image)[0]
#
# d = "images/Graham.jpg"
# Graham_image = face_recognition.load_image_file(d)
# Graham_face_encoding = face_recognition.face_encodings(Graham_image)[0]

# e = "images/Anurag.jpg"
# Anurag_image = face_recognition.load_image_file(e)
# Anurag_face_encoding = face_recognition.face_encodings(Anurag_image)[0]
#
# f = "images/Lucia.jpg"
# Lucia_image = face_recognition.load_image_file(f)
# Lucia_face_encoding = face_recognition.face_encodings(Lucia_image)[0]
#
# g = "images/Blake.jpg"
# Blake_image = face_recognition.load_image_file(g)
# Blake_face_encoding = face_recognition.face_encodings(Blake_image)[0]
#
# h = "images/Petrus.jpg"
# Petrus_image = face_recognition.load_image_file(h)
# Petrus_face_encoding = face_recognition.face_encodings(Petrus_image)[0]
#
# i = "images/Fiona.jpg"
# Fiona_image = face_recognition.load_image_file(i)
# Fiona_face_encoding = face_recognition.face_encodings(Fiona_image)[0]
#
# j = "images/Francine.jpg"
# Francine_image = face_recognition.load_image_file(j)
# Francine_face_encoding = face_recognition.face_encodings(Francine_image)[0]
#
# k = "images/Sid.jpg"
# Sid_image = face_recognition.load_image_file(k)
# Sid_face_encoding = face_recognition.face_encodings(Sid_image)[0]
#
# l = "images/Kevin.jpg"
# Kevin_image = face_recognition.load_image_file(l)
# Kevin_face_encoding = face_recognition.face_encodings(Kevin_image)[0]
#
# m = "images/Laine.jpg"
# Laine_image = face_recognition.load_image_file(m)
# Laine_face_encoding = face_recognition.face_encodings(Laine_image)[0]
#
# n = "images/Mark.jpg"
# Mark_image = face_recognition.load_image_file(n)
# Mark_face_encoding = face_recognition.face_encodings(Mark_image)[0]
#
# o = "images/Noaman.jpg"
# Noaman_image = face_recognition.load_image_file(o)
# Noaman_face_encoding = face_recognition.face_encodings(Noaman_image)[0]
#
# p = "images/Ranil.jpg"
# Ranil_image = face_recognition.load_image_file(p)
# Ranil_face_encoding = face_recognition.face_encodings(Ranil_image)[0]
#
# q = "images/William.jpg"
# William_image = face_recognition.load_image_file(q)
# William_face_encoding = face_recognition.face_encodings(William_image)[0]
#
# r = "images/Suresh.jpg"
# Suresh_image = face_recognition.load_image_file(r)
# Suresh_face_encoding = face_recognition.face_encodings(Suresh_image)[0]
#
# s = "images/Trista.jpg"
# Trista_image = face_recognition.load_image_file(s)
# Trista_face_encoding = face_recognition.face_encodings(Trista_image)[0]
#
# t = "images/Dan.jpg"
# Dan_image = face_recognition.load_image_file(t)
# Dan_face_encoding = face_recognition.face_encodings(Dan_image)[0]
#
# u = "images/Peter.jpg"
# Peter_image = face_recognition.load_image_file(u)
# Peter_face_encoding = face_recognition.face_encodings(Peter_image)[0]
#
# v = "images/James.jpg"
# James_image = face_recognition.load_image_file(v)
# James_face_encoding = face_recognition.face_encodings(James_image)[0]
#
# w = "images/Satyam.jpg"
# Satyam_image = face_recognition.load_image_file(w)
# Satyam_face_encoding = face_recognition.face_encodings(Satyam_image)[0]
#
# x = "images/RamC.jpg"
# RamC_image = face_recognition.load_image_file(x)
# RamC_face_encoding = face_recognition.face_encodings(RamC_image)[0]
#
# y = "images/Betty.jpg"
# Betty_image = face_recognition.load_image_file(y)
# Betty_face_encoding = face_recognition.face_encodings(Betty_image)[0]
#
# z = "images/Greg.jpg"
# Greg_image = face_recognition.load_image_file(z)
# Greg_face_encoding = face_recognition.face_encodings(Greg_image)[0]
#
# aa = "images/Linda.jpg"
# Linda_image = face_recognition.load_image_file(aa)
# Linda_face_encoding = face_recognition.face_encodings(Linda_image)[0]
#
# ab = "images/Richard.jpg"
# Richard_image = face_recognition.load_image_file(ab)
# Richard_face_encoding = face_recognition.face_encodings(Richard_image)[0]
#
# ac = "images/Tara.jpg"
# Tara_image = face_recognition.load_image_file(ac)
# Tara_face_encoding = face_recognition.face_encodings(Tara_image)[0]
#
# ad = "images/RamakrishnanS.jpg"
# RamakrishnanS_image = face_recognition.load_image_file(ad)
# RamakrishnanS_face_encoding = face_recognition.face_encodings(RamakrishnanS_image)[0]
#
# ae = "images/Jessica.jpg"
# Jessica_image = face_recognition.load_image_file(ae)
# Jessica_face_encoding = face_recognition.face_encodings(Jessica_image)[0]
#
# af = "images/Olivia.jpg"
# Olivia_image = face_recognition.load_image_file(af)
# Olivia_face_encoding = face_recognition.face_encodings(Olivia_image)[0]
#
# ag = "images/Olivia.jpg"
# Olivia_image = face_recognition.load_image_file(ag)
# Olivia_face_encoding = face_recognition.face_encodings(Olivia_image)[0]
#
# ah = "images/Grace.jpg"
# Grace_image = face_recognition.load_image_file(ah)
# Grace_face_encoding = face_recognition.face_encodings(Grace_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    # soum_face_encoding
    # Yusuf_face_encoding,
    # sarit_face_encoding,
    # Ben_face_encoding
    # Graham_face_encoding,
    # Anurag_face_encoding
    # Lucia_face_encoding,
    # Blake_face_encoding,
    # Petrus_face_encoding,
    # Fiona_face_encoding,
    # Francine_face_encoding,
    # Sid_face_encoding,
    # Kevin_face_encoding,
    # Laine_face_encoding,
    # Mark_face_encoding,
    # Noaman_face_encoding,
    # Ranil_face_encoding,
    # William_face_encoding,
    # Suresh_face_encoding,
    # Trista_face_encoding,
    # Dan_face_encoding,
    # Peter_face_encoding,
    # James_face_encoding,
    # Satyam_face_encoding,
    # RamC_face_encoding,
    # Betty_face_encoding,
    # Greg_face_encoding,
    # Linda_face_encoding,
    # Richard_face_encoding,
    # Tara_face_encoding,
    # RamakrishnanS_face_encoding,
    # Jessica_face_encoding,
    # Olivia_face_encoding,
    # Grace_face_encoding


]
known_face_names = [
    # "Soum"
    # "Yusuf",
    # "Sarit",
    # "Ben"
    #  "Graham",
    # "Anurag"
    # "Lucia",
    # "Blake",
    # "Petrus",
    # "Fiona",
    # "Francine",
    # "Sid",
    # "Kevin",
    # "Laine",
    # "Mark",
    # "Noaman",
    # "Ranil",
    # "William",
    # "Suresh",
    # "Trista",
    # "Dan",
    # "Peter",
    # "James",
    # "Satyam",
    # "RamC",
    # "Betty",
    # "Greg",
    # "Linda",
    # "Richard",
    # "Tara",
    # "RamakrishnanS",
    # "Jessica",
    # "Olivia",
    # "Grace"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def face_compare(frame,process_this_frame):
    print ("compare")
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    return face_names
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        #cv2.rectangle(frame, (left, bottom+36), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+20), font, 0.3, (255, 255, 255), 1)
        print ("text print")

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./test/testvdo.mp4') # Video file source

while cap.isOpened(): # True:
    ret, frame = cap.read()

    #frame = video_capture.read()[1]

    # To print the facial landmarks
    # landmrk = face_recognition.face_landmarks(frame)
    # for l in landmrk:
    #     for key,val in l.items():
    #         for (x,y) in val:
    #             cv2.circle(frame, (x, y), 1, (255,0, 0), -1)


    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)
    # face_locations = face_recognition.face_locations(rgb_image)
    # print (reversed(face_locations))
    face_name = face_compare(rgb_image,process_this_frame)
    for face_coordinates, fname in zip(faces,face_name):
        print ("forrrrr")
        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue


        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        if fname == "Unknown":
            name = emotion_text
        else:
            name = str(fname) + " is " + str(emotion_text)
        
        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, name,
                  color, 0, -45, 1.5, 1)


    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

