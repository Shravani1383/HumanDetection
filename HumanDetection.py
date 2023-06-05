#task 1
#Import the necessary libraries, including OpenCV
import cv2
# Initialize the laptop camera using OpenCV.
cam = cv2.VideoCapture(0)
#task 2
# Load the pre-trained human detection model, such as the OpenCV HOG-based pedestrian detector
hog = cv2.HOGDescriptor()
# Load the model using OpenCV
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#task 3
# Enter a loop to continuously read frames from the laptop camera
while True:
    # Read frames from the camera
    ret, frame = cam.read()
    
    # Convert each frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply human detection model
    boxes, scores = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
#task 4
    # Iterate over the detected bounding boxes and confidence scores.
    for (a, b, c, d) in boxes:
        cv2.rectangle(frame, (a, b), (a + b, c + d), (0, 255, 0), 2)

    # Display frame with bounding boxes
    cv2.imshow('Human Detection', frame)
#task 5
    # Calculate the total number of humans detected in each frame.
    total_humans = len(boxes)
    
    # Print the count of detected humans on the console.
    print("Number of Humans Detected:", total_humans)
#task 6
    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera
cam.release()
cv2.destroyAllWindows()

