from geometry_msgs.msg import Point

import numpy as np

def pointmsg_to_numpyarray(msg, _3d=False):
    if _3d:
        return np.array([msg.x, msg.y, msg.z])

    return np.array([msg.x, msg.y])