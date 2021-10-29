import matplotlib.pyplot as plt

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
def convert_display_to_data_coordinates(transData, length=10):
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in x axis
    data_coords = inv.transform([(0, 0), (length, 0)])
    # get the length of the segment in data units
    x_coord_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    y_coord_len = data_coords[1, 1] - data_coords[0, 1]
    return x_coord_len, y_coord_len


# range of x and y axis
xmin_ax = -0.1
xmax_ax = 1.1
ymin_ax = -0.3
ymax_ax = 1.3

# font size
fontsize = 16
# arrows head length and head width
hl = 10
hw = 6

fig = plt.figure(0, figsize=(5, 3), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=12)


# x vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(1, 0), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# y vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0.6, 1), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# y-ax vector
plt.annotate("", xytext=(0.6, 0), xycoords='data', xy=(0.6, 1), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# x vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0.6, 0), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# perpendicular symbol
plt.plot([0.6+htl, 0.6+htl], [0, vtl], 'k', lw=1)
plt.plot([0.6, 0.6+htl], [vtl, vtl], 'k', lw=1)

x_bl = -0.15
# labels
plt.text(0.6, x_bl, r'$a\mathbf{x}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(1, x_bl, r'$\mathbf{x}$', fontsize=fontsize, ha='center', va='baseline')

plt.text(0.55, 1, r'$\mathbf{y}$', fontsize=fontsize, ha='right', va='center')
plt.text(0.63, 0.9, r'$\mathbf{y}-a\mathbf{x}$', fontsize=fontsize, ha='left', va='center')

plt.text(-0.05, 1, r'$(\mathbf{y}-a\mathbf{x}) \perp \mathbf{x}$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('mse_linear_homogeneous.pdf', bbox_inches='tight')
plt.show()


