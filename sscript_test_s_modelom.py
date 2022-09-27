#************PROJEKTNI ZADATAK**********************#
#                                                   #
#                  Designed by:                     #
#      Ivan Gudelj - 1. godina diplomskog studija   #
#           Robotika i umjetna inteligencija        #
#                                                   #
#***************************************************#
#                                                   #
#                   Kolegij:                        #
#               Meko racunarstvo                    #
#   LED SEGMENTS CONTROLED BY HAND GESTURES         #
#                                                   #
#***************************************************#

# Import necessary packages
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import requests
import time
import face_recognition
import os
import glob

# Enable or disable requests (might crash if the segments arent connected to the internet, default = False)
# Set the IpAddresses of each segment
requestsEnabled = False
ipAddress = 'http://192.168.108.242/'
ipAddress1 = 'http://192.168.108.242/'
ipAddress2 = 'http://192.168.108.193/'
sendRequestToFirstSegment = True


def sendHTTPRequest(className):
    control = ''
    if className == 'thumbs up':
        response = requests.get(ipAddress + 'win&A=~10')
        control = 'Turn Up Brigthness'
        time.sleep(0.5)
    elif className == 'thumbs down':
        response = requests.get(ipAddress + 'win&A=~-10')
        control = 'Turn Down Brigthness'
        time.sleep(0.5)
    elif className == 'stop' or className == 'live long':
        response = requests.get(ipAddress + 'win&FX=0')
        control = 'Set to Solid Color'
        time.sleep(1)
    elif className == 'rock':
        response = requests.get(ipAddress + 'win&RB')
        control = 'Reboot Device'
        time.sleep(2)
    elif className == 'call me':
        response = requests.get(ipAddress + 'win&FX=~2')
        control = 'Change Effect'
        time.sleep(1)
    if control != '':
        print(' Wanted Control Recognized: ' + control)

# Initialize mediapipe and control variables
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.55)
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
model.summary()

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

#############################################################################################
#       FACE RECOGNITION PART
# Make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'known_people/')

# Make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob(path+'*.jpg')]
# Find number of known faces
number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

    # Create array of known names
    names[i] = names[i].replace(dirname+"/known_people/", "")  
    known_face_names.append(names[i])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
#############################################################################################


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    name = ''
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

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if name != "Unknown":
            cv2.putText(frame, name[56:66], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #requestsEnabled = True
        else:
            requestsEnabled = False
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Check if user is authorized to run the commands (Currently Hardcoded)
    if not name =="Unkown" and name[56:66] == "IvanGudelj":
        x, y, c = frame.shape

        # Flip the frame vertically
        #frame = cv2.flip(frame, 1)

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)
        className = ''

        # Post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                    mpDrawingStyles.get_default_hand_landmarks_style(), 
                                    mpDrawingStyles.get_default_hand_connections_style())
                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

                # 'Peace' gesture changes control to other segment (can be added more)
                if className == 'peace':
                    sendRequestToFirstSegment = not sendRequestToFirstSegment
                    print('Please wait few seconds')
                    print('Control switched to: ' + ipAddress)
                    time.sleep(2)
                # Change of segments to control
                if sendRequestToFirstSegment:
                    ipAddress = ipAddress1
                else:
                    ipAddress = ipAddress2
                
                # Send request over HTTP
                if requestsEnabled :
                    sendHTTPRequest(className)
        # Show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255), 2, cv2.LINE_AA)
        # Show the final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()