from ultralytics import YOLO    # YOLOv8 --> object detection model from Ultralytics
import cv2                      # OpenCV --> reading/writing video, 
                                # drawing boxes (important for player detection)
                                # color conversions, etc.

model = YOLO("yolov8n.pt")      # this is the AI model
                                # nano model yolov8(n) --> smallest and fastest but less accurate
                                #   they also have yolov8(s) and yolov8(m) which scale in size
                                # good for prototyping, less heavy on the CPU 

# open the video file
video_path = "dortmundsnippet.mp4.webm"
cap = cv2.VideoCapture(video_path)      # returns a VideoCapture object that I named 'cap'
                                        # use cap.read() in the loop to get each frame

# get properties of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))                # frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # width of video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # height of video (in pixels)

# Output Writer
# create an object that we can write frames to
# we need the properties to align so that the output doesn't get distorted or clipped
#   by this i mean fps, width, and height
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) 

while True:
    ret, frame = cap.read()           # returns the frames of the video
    if not ret:                       # ret is a boolen; returns whether or not the frame was read successfully
        break                         # break in case the video ends or error

    # run the yolo model on the video
    results = model(frame, verbose=False)[0]        # run the model on the video!
                                                    # the model returns a list of all objects - one per input frame
                                                    #   since we're going frame by frame, it just returns the frame we're on
                                                    # results contains:
                                                    #   detection boxes
                                                    #   class ids
                                                    #   confidences
                                                    #   masks
                                                    #   any other metadata

    for box in results.boxes:                                       # container of detection boxes (within a single frame)
        cls = int(box.cls[0])                                       # class ids for the box, converted to an int
        if cls == 0:                                                # ID = 0 for people, will have to introduce/train
                                                                    # ball detection later
            x1, y1, x2, y2 = map(int, box.xyxy[0])                  # coordinates of the bounding box (returned as a 2D array)
                                                                    # and then we map them to the pixels
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)    # draw the box, green, with border of 2

            conf = float(box.conf[0])                               # confidence score of box
            name = model.names[cls]                                 # string label for box
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


    out.write(frame)                        # write the frame to the output video
    cv2.imshow("Soccer Analysis", frame)    # live window titled soccer analysis

    if cv2.waitKey(1) & 0xFF == ord('q'):   # if 'q' is pressed while output video is running, quit the loop
        break                               # and end the program

cap.release()                               # wrap up and close all programs
out.release()
cv2.destroyAllWindows()