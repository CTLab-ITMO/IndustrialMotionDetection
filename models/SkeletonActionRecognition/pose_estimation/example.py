import cv2

from pose_estimation.inference_pose import PoseEstimator

# Пути к папкам
input_path = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/data/frames2/frame_0097.jpg'
output_path = '/Users/anastasiaspileva/Desktop/people_on_roof.jpg'


estimator = PoseEstimator("yolo11m-pose.pt")
estimator.convert_to_openvino("yolo11m-pose")
estimator.predict_openvino("CPU", input_path)
result = estimator.predict(input_path)
cv2.imwrite(output_path, result)
