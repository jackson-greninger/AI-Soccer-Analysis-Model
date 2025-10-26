from ultralytics import YOLO    # YOLOv8 --> object detection model from Ultralytics
import cv2                      # OpenCV --> reading/writing video, 
                                # drawing boxes (important for player detection)
                                # color conversions, etc.
import torch
import numpy as np
import Util

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8m.pt").to(device)       # this is the AI model
                                            #   -(n) nano, (s) small, (m) medium, (l) large, (x) xtra large 

# open the video file
video_path = "videos/netherlands_us_wwc.mp4"
cap = cv2.VideoCapture(video_path)      

prev_frame = None
frame_count = 0

last_ball_center = 0        # variables for optical flow tracking
missed_frames = 0           # when we can't see the ball, where will it be next?

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

    ####################
    # Green Field Mask #
    ####################
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35,40,40])
    upper_green = np.array([85,255,255])
    field_mask = cv2.inRange(hsv, lower_green, upper_green)

    combined_mask = cv2.bitwise_and(thresh, field_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    ####################
    # Finding The Ball #
    ####################
    ball_candidate = None
    h, w = frame.shape[:2]

    # adaptive thresholds
    min_ball_area = max(5, (w*h) * 0.00001)
    max_ball_area = (w*h) * 0.001
    
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_ball_area or area > max_ball_area:
            continue

        (x, y, bw, bh) = cv2.boundingRect(contour)
        aspect_ratio = bw / float(bh)

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / perimeter ** 2)

        if 0.7 < circularity < 1.2 and 0.8 < aspect_ratio < 1.2:
            ball_candidate = (x, y, bw, bh)
            break

    ######################
    # Optical Flow Logic #
    ######################
    if ball_candidate:
        (x, y, bw, bh) = ball_candidate
        ball_center = (int(x + bw/2), int(y + bh/2))
        last_ball_center = ball_center
        missed_frames =0

        # draw the circle on the frame
        cv2.circle(frame, last_ball_center, 6, (0,165,255), -1)
        Util.safe_draw_text(frame, "Predicted Ball", (last_ball_center[0], last_ball_center[1] - 10), (0,165,255), 0.5, 1)

    ####################
    # Player Detection #
    ####################
    frame_count += 1

    if frame_count % 2 == 0:  # every other frame (easier on the GPU)
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                conf = float(box.conf[0])
                cv2.putText(frame, f"Player {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    ###########
    # Display #
    ###########
    cv2.putText(frame, "Players (Green) | Ball (Red) | Predicted (Orange)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    
    cv2.imshow("Soccer Analysis", frame)

    prev_frame = gray_blur

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()                   # wrap up and close all programs
cv2.destroyAllWindows()