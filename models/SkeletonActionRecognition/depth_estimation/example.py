import cv2
import matplotlib.pyplot as plt
from depth_estimation import DepthEstimator

device = 'cpu'
model_path = '/depth_estimation/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
raw_img = cv2.imread('/data/frames2/frame_0096.jpg')
estimator = DepthEstimator(model_config='vits', model_path=model_path, device=device)
depth = estimator.predict(raw_img)
print(depth)

plt.imshow(depth)
plt.axis('off')
plt.savefig('/Users/anastasiaspileva/Desktop/depth_on_stairs.jpg')
