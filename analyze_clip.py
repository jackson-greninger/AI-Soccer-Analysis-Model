from ultralytics import YOLO    # YOLOv8 --> object detection model from Ultralytics
import cv2                      # OpenCV --> reading/writing video, 
                                # drawing boxes (important for player detection)
                                # color conversions, etc.
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8m.pt").to(device)       # this is the AI model
                                            # nano model yolov8(n) --> smallest and fastest but less accurate
                                            #   they also have yolov8(s) and yolov8(m) which scale in size
                                            # good for prototyping, less heavy on the CPU 

# open the video file
video_path = "videos/netherlands_us_wwc.mp4"
cap = cv2.VideoCapture(video_path)      # returns a VideoCapture object that I named 'cap'
                                        # use cap.read() in the loop to get each frame
prev_frame = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    ##################
    # PRE-PROCESSING #
    ##################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)     # can change the tuple to adjust for blur radius and accuracy

    if prev_frame is None:
        prev_frame = gray_blur
        continue

    delta = cv2.absdiff(prev_frame, gray_blur)

    thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)[1]    # tweak second parameter to adjust for motion size    
    
    # clean the noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_candidate = None

    # Define acceptable contour area range for ball-like objects
    min_ball_area, max_ball_area = 5, 40  # <-- tweak these for your resolution
    
    for contour in contours:

        area = cv2.contourArea(contour)
        if area < min_ball_area or area > max_ball_area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / perimeter ** 2)

        if 0.6 < circularity < 1.3 and 0.8 < aspect_ratio < 1.2:
            ball_candidate = (x, y, w, h)
            break

    if ball_candidate:
        (x, y, w, h) = ball_candidate
        ball_center = (int(x + w/2), int(y + h/2))
        cv2.circle(frame, ball_center, 6, (0,0,255), -1)
        cv2.putText(frame, "Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # --- YOLO (for players) ---
    frame_count += 1

    if frame_count % 2 == 0:  # optional: every other frame
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                conf = float(box.conf[0])
                cv2.putText(frame, f"Player {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- Display ---
    cv2.putText(frame, "Players (Green) | Ball (Red)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("Soccer Analysis", frame)

    prev_frame = gray_blur

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()                               # wrap up and close all programs
cv2.destroyAllWindows()