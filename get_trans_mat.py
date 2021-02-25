import bpy
import mathutils
import numpy as np


def get_trans_mat():
    if bpy.context.object:
        mat = np.array(bpy.context.object.matrix_local)
        np.save("C:\\Projects\\bvh_tool\\data\\transformation.npy", mat)


if __name__ == '__main__':
    get_trans_mat()
