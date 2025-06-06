import cv2
from ultralytics import YOLO
import vlc
import time

#trained model
model = YOLO('best-yolo11.pt')

player = vlc.MediaPlayer("assets/alert.mp3")

video = cv2.VideoCapture("assets/video7.mov")

last_play_time = 0
play_interval = 10

def notify():
    global last_play_time
    current_time = time.time()

    if current_time - last_play_time >= play_interval:
        player.stop()
        player.play()
        last_play_time = current_time

ret = True


CONFIDENCE_THRESHOLD = 0.4

while ret:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf.item() 
            #skip when confidence score is low
            if conf < CONFIDENCE_THRESHOLD:
                continue 

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]

            color = (0, 255, 0) if label == "life_jacket" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            if label == "no_life_jacket":
                notify()

    cv2.imshow("Lifejacket Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()