import numpy as np
import matplotlib.pyplot as plt
import cv2

orig = cv2.imread('disp_orig.png', cv2.IMREAD_ANYDEPTH).astype(np.float32)
ta = cv2.imread('disp_ta.png', cv2.IMREAD_ANYDEPTH).astype(np.float32)
es = cv2.imread('disp_es.png', cv2.IMREAD_ANYDEPTH).astype(np.float32)


plt.figure()
plt.subplot(2,2,1); plt.imshow(orig / 16, vmin=0, vmax=4, cmap='magma')
plt.subplot(2,2,2); plt.imshow(ta / 16, vmin=0, vmax=4, cmap='magma')
plt.subplot(2,2,3); plt.imshow(es / 16, vmin=0, vmax=4, cmap='magma')
plt.subplot(2,2,4); plt.imshow(np.abs(es - ta) / 16, vmin=0, vmax=1, cmap='magma')
plt.show()
