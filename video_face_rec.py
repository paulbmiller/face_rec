import face_recognition
import os
import cv2
import pickle
import dlib

dlib.DLIB_USE_CUDA = True

# https:\\\\www.youtube.com\\watch?v=PdkPI92KSIs

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

video = cv2.VideoCapture('obamaspeech.mp4')

"""
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
"""

print("Loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}\\{name}"):
        image = face_recognition.load_image_file(
            f"{KNOWN_FACES_DIR}\\{name}\\{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        
        known_faces.append(encoding)
        known_names.append(name)

nb_known = len(os.listdir(KNOWN_FACES_DIR))
if nb_known > 0:
    next_id = nb_known
else:
    next_id = 0

start_frame_number = 650
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)


print("Processing unknown faces")
while True:
    RATIO = 1
    ret, original = video.read()
    
    original = original[:-80, 160:]
    
    image = cv2.resize(original, (original.shape[1] // RATIO,
                                  original.shape[0] // RATIO))

    # image = face_recognition.load_image_file()
    locations = face_recognition.face_locations(image,
                                                number_of_times_to_upsample=1,
                                                model=MODEL)

    encodings = face_recognition.face_encodings(image, locations)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding,
                                                 TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            """
            os.mkdir(f"{KNOWN_FACES_DIR}\\{match}")
            file = open(
                f"{KNOWN_FACES_DIR}\\{match}\\{match}-{int(time.time())}.pkl",
                "wb")
            pickle.dump(face_encoding, file)
            """

        top_left = (face_location[3] * RATIO, face_location[0] * RATIO)
        bottom_right = (face_location[1] * RATIO, face_location[2] * RATIO)
        color = [0, 255, 0]
        cv2.rectangle(original, top_left, bottom_right, color,
                      FRAME_THICKNESS)

        top_left = (face_location[3] * RATIO, face_location[2] * RATIO)
        bottom_right = (face_location[1] * RATIO,
                        face_location[2] * RATIO + 22)
        cv2.rectangle(original, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(original, str(match),
                    (face_location[3] * RATIO + 10,
                     face_location[2] * RATIO + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                    FONT_THICKNESS)

    cv2.imshow("", original)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
