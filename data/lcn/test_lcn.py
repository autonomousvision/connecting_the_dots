import numpy as np
import matplotlib.pyplot as plt
import lcn
from scipy import misc

# load and convert to float
img = misc.imread('img.png')
img = img.astype(np.float32)/255.0

# normalize
img_lcn, img_std = lcn.normalize(img,5,0.05)

# normalize to reasonable range between 0 and 1
#img_lcn = img_lcn/3.0
#img_lcn = np.maximum(img_lcn,0.0)
#img_lcn = np.minimum(img_lcn,1.0)

# save to file
#misc.imsave('lcn2.png',img_lcn)

print ("Orig Image: %d x %d (%s), Min: %f, Max: %f" % \
(img.shape[0], img.shape[1], img.dtype, img.min(), img.max()))
print ("Norm Image: %d x %d (%s), Min: %f, Max: %f" % \
(img_lcn.shape[0], img_lcn.shape[1], img_lcn.dtype, img_lcn.min(), img_lcn.max()))

# plot original image
plt.figure(1)
img_plot = plt.imshow(img)
img_plot.set_cmap('gray')
plt.clim(0, 1) # fix range
plt.tight_layout()

# plot normalized image
plt.figure(2)
img_lcn_plot = plt.imshow(img_lcn)
img_lcn_plot.set_cmap('gray')
#plt.clim(0, 1) # fix range
plt.tight_layout()

# plot stddev image
plt.figure(3)
img_std_plot = plt.imshow(img_std)
img_std_plot.set_cmap('gray')
#plt.clim(0, 0.1) # fix range
plt.tight_layout()

plt.show()
