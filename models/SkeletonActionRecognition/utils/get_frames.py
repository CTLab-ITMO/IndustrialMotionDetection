import cv2
import os

# Путь к видеофайлу
video_path = '/content/video2.mp4'

# Директория для сохранения кадров
output_dir = '/content/frames2'
os.makedirs(output_dir, exist_ok=True)

# Открываем видео
cap = cv2.VideoCapture(video_path)

# Проверяем, удалось ли открыть видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

frame_count = 0

while True:
    # Читаем кадр из видео
    ret, frame = cap.read()

    # Если кадр не удалось прочитать, выходим из цикла
    if not ret:
        break

    # Сохраняем кадр в файл
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, cv2.flip(frame, 0))
    print(f'Сохранен кадр: {frame_filename}')
    frame_count += 1

# Освобождаем ресурсы
cap.release()
print(f'Всего сохранено кадров: {frame_count}')
