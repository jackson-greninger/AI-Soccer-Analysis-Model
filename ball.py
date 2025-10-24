import numpy as np

# Add this before the loop:
prev_frame = None

# Inside the main loop, after reading `frame`:
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

if prev_frame is None:
    prev_frame = gray
    continue

# Frame difference for motion detection
delta = cv2.absdiff(prev_frame, gray)
thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

# Find moving regions
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) < 300:  # ignore small noise
        continue
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

prev_frame = gray