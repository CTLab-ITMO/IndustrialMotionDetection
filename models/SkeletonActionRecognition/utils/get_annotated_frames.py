import cv2
import os


def get_annoted_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        exit()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, cv2.flip(frame, 0))
        print(f'Сохранен кадр: {frame_filename}')
        frame_count += 1
    cap.release()
    print(f'Всего сохранено кадров: {frame_count}')


video_path = '/data/video2.mp4'
output_dir = '/data/frames2'
os.makedirs(output_dir, exist_ok=True)
get_annoted_frames(video_path, output_dir)
