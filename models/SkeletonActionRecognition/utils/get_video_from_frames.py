import cv2
import os

frames_folder = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/yolo/yolo_video2/'
frames = sorted([os.path.join(frames_folder, img) for img in os.listdir(frames_folder) if
                 img.endswith(".png") or img.endswith(".jpg")])

if not frames:
    print("Нет кадров в указанной папке.")
    exit()

first_frame = cv2.imread(frames[0])
height, width, layers = first_frame.shape

fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
output_video = cv2.VideoWriter('/content/annotated_video2.mp4',
                               fourcc, 30, (width, height))

for frame in frames:
    img = cv2.imread(frame)
    output_video.write(img)

output_video.release()
print("Видео успешно создано!")
