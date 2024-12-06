import os
import numpy as np
import cv2
from numpy import loadtxt
from tensorflow import keras
# import dlib

# loading the expression model
# Load the JSON file containing the model architecture
with open('network_emotions.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model from the JSON
model = keras.models.model_from_json(loaded_model_json)

# Load the weights into the model
model.load_weights('network_emotions.weights.h5')
print("Loaded emotion model from disk")


cap = cv2.VideoCapture(0)

# load face detection model
face_model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image_number=0
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
predictions = []

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


print("Press 's' to save a frame, 'q' to quit.")

while True:
    # Capture each frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # # Display the frame in a window
    # cv2.imshow("Camera Feed", frame)

#   yaha se squrare banne ka start hoga


    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_model.detectMultiScale(gray_frame,scaleFactor=1.5, minNeighbors=3)
    for (i, rect) in enumerate(faces):
        x,y,w,h=rect
        #draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Extract the face region from the original image
        face_roi = gray_frame[y:y + h, x:x + w]

         # Resize the face ROI to match the model's input shape (48x48)
        resized_face = cv2.resize(face_roi, (48, 48))

         # Reshape and normalize the face ROI to match the model's input format
         # Assuming your model expects a single channel (grayscale) image
        resized_face = resized_face.reshape(1, 48, 48, 1) / 255.0

        # Make prediction on the extracted and preprocessed face
        prediction = model.predict(resized_face)
        predictions.append(prediction) # Append to prediction list
        prediction = np.argmax(prediction)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emotion_labels[prediction], (x, y - 10), font, 0.5, (0,255,0), 1)





    # Display the frame in a window
    cv2.imshow("Camera Feed", frame)
    # Wait for a key press
    key = cv2.waitKey(1)

    # Save the frame if 's' is pressed
    if key == ord('s'):
        filename=f"image{image_number}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as 'image{filename}.jpg'.")
        image_number+=1

    # Break the loop if 'q' is pressed
    elif key == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

