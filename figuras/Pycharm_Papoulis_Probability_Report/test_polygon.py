import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-130, 130)
    ax.set_ylim(-130, 130)
    ax.set_zlim(0, 100)
    return fig


# Red triangle
T1 = [[-55.9, 47.0, 50],
      [23.2, 52.5, 50],
      [-7.3, -55.4, 50]]

# Blue triangle
T2 = [[81.3, -54.1, 50],
      [87.7, -96.6, 50],
      [60.6, -96.9, 50]]

# Green triangle
T3 = [[68.6, -64.2, 70],
      [31.3, -64.8, 70],
      [51.8, -23.3, 70]]


# Add all triangles in a single collection. This works as expected.
fig = create_figure()
coll = Poly3DCollection([T1, T2, T3], facecolors=['r', 'b', 'g'], edgecolors=['r', 'b', 'g'])
fig.gca().add_collection(coll)
fig.gca().set_title("Correct behaviour")


# Add triangles in two separate collections. This exposes a bug where
# the green triangle is displayed underneath the red one even though
# it should be on top of it. Note, however, that if we omit the blue
# triangle (T2) from the first collection then the bug disappears.
fig = create_figure()
coll1 = Poly3DCollection([T1, T2], facecolors=['r', 'b'], edgecolors=['r', 'b'])
coll2 = Poly3DCollection([T3], facecolors=['g'], edgecolors=['g'])
fig.gca().add_collection(coll1)
fig.gca().add_collection(coll2)
fig.gca().set_title("Buggy behaviour (but works if blue triangle is removed)")

plt.show()