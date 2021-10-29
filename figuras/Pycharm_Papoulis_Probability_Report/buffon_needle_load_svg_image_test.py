import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as colors
from matplotlib import transforms
from scipy import ndimage

from matplotlib import cm
from matplotlib import rc


__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


im = 'buffon_needle_2.png'
img = plt.imread(im)
fig, ax = plt.subplots()
ax.set_xlim(0, 60)
ax.set_ylim(0, 40)

# compute angle
theta = math.atan(img.shape[0]/img.shape[1])
print(theta)


frame_height = 25
frame_width = frame_height * img.shape[1] / img.shape[0]
x_start = 20
y_start = 15
ax.imshow(img, extent=[x_start, x_start + frame_width, y_start, y_start + frame_height])
ax.imshow(img, extent=[x_start + 10, x_start + frame_width + 10, y_start - 10, y_start+frame_height - 10])

#rotated_img = ndimage.rotate(img, -theta * 180 / math.pi, reshape=True, order=1)
#print(rotated_img.shape)
#plt.imshow(rotated_img)
#ax.imshow(rotated_img, extent=[x_start, x_start + frame_width, y_start, y_start + frame_height])

tr = transforms.Affine2D().rotate_deg(-30)
tr1 = transforms.Affine2D().rotate_deg_around(10, 10, -30)
tr2 = transforms.Affine2D().rotate_deg_around(10, 10, -180 * theta / math.pi)
#tr = transforms.Affine2D().rotate_deg(-180 * theta / math.pi)
#ax.imshow(img, transform=tr)
ax.imshow(img, extent=[10, 10 + frame_width, 10, 10+frame_height])
ax.imshow(img, transform=tr1 + ax.transData, extent=[10, 10 + frame_width, 10, 10+frame_height])
ax.imshow(img, transform=tr2 + ax.transData, extent=[10, 10 + frame_width, 10, 10+frame_height])

plt.plot([0, 60], [10, 10], 'k')

#ax.imshow(img, transform=tr, extent=[x_start, x_start + frame_width, y_start, y_start + frame_height])


plt.savefig('buffon_needle_load_svg_image_test.pdf', bbox_inches='tight', dpi=900)
plt.show()
