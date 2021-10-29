import matplotlib.pyplot as plt

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


# range of x and y axis
xmin_ax = -1
xmax_ax = 2
ymin_ax = -0.75
ymax_ax = 1

# font size
fontsize = 16
# arrows head length and head width
hl = 10
hw = 6
hl_ax = 8
hw_ax = 4

fig = plt.figure(0, figsize=(5, 3), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# x axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# z axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# y axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, -0.6), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))

# x1 vector
x1_x = 1.8
x1_y = 0
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# x2 vector
x2_x = -0.5
x2_y = -0.75
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2_x, x2_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

s_x = 1
s_y = 0.75
# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# s projection
sp_x = s_x
sp_y = -0.6
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(sp_x, sp_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# sp over x1 projection
x1p_x = (x2_x/x2_y) * (x1_y - sp_y) + sp_x
x1p_y = 0
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1p_x, x1p_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
plt.plot([sp_x, x1p_x], [sp_y, x1p_y], 'k--', lw=1)

# sp over x2 projection
x2p_x = (x2_x/x2_y) * sp_y
x2p_y = sp_y
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2p_x, x2p_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
plt.plot([sp_x, x2p_x], [sp_y, x2p_y], 'k--', lw=1)

# sp to s arrow
plt.annotate("", xytext=(sp_x, sp_y), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))


# labels
plt.text(x1_x, x1_y+0.05, r'$\mathbf{x_1}$', fontsize=fontsize, ha='right', va='bottom')
plt.text(x1p_x+0.1, x1p_y+0.05, r'$a_1\mathbf{x_1}$', fontsize=fontsize, ha='right', va='bottom')

plt.text(x2_x-0.04, x2_y, r'$\mathbf{x_2}$', fontsize=fontsize, ha='right', va='center')
plt.text(x2p_x-0.01, x2p_y, r'$a_2\mathbf{x_2}$', fontsize=fontsize, ha='right', va='bottom')

plt.text(sp_x/2-0.15, sp_y/2, r'$\hat{\mathbf{s}}$', fontsize=fontsize, ha='right', va='center')
plt.text(s_x/2-0.15, s_y/2, r'$\mathbf{s}$', fontsize=fontsize, ha='center', va='center')

plt.text(s_x+0.04, s_y/2+0.1, r'$\varepsilon$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('mse_linear_general_case.pdf', bbox_inches='tight')
plt.show()


