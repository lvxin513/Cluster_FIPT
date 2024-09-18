import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

path = "/proj/users/xlv/lvxin/fipt/data/kitchen/specular/000_0_0.exr"

img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
img = img ** (1/2.2)
img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Convert the normalized image to 8-bit (0-255)
img= (img * 255).astype(np.uint8)
cv2.imwrite('1.png',img)