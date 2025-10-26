import cv2

def safe_draw_text(img, text, pos, color, font_scale=0.5, thickness=1):
    x, y = pos
    h, w = img.shape[:2]
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)