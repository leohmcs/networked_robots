import numpy as np

# rotation around z-axis
def Rz(x, y, z, th):
    mat = np.array([[np.cos(th), np.sin(th), 0], [np.cos(th), -np.sin(th), 0], [0, 0, 1]])
    p = np.array([x, y, z])

    return np.dot(mat, p)