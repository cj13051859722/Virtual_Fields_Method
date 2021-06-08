import numpy as np
import scipy.io as scio


def real_epsilon_fig2_7(P, E, nu, L, l, t, Step, mmPerPix):
    G = E / (2 * (1 + nu))
    # Dimensions
    c = l / 2
    I = t * l ** 3 / 12  # Second moment of area respect to z-axis mm^4
    # CCD and 2D-DIC parameters
    mesh_size = Step * mmPerPix
    x = np.arange(mesh_size / 2, L, mesh_size)
    y = np.arange(c - mesh_size / 2, -c, -mesh_size)
    # Initial positions
    x_Mat = np.zeros((len(y), len(x)), dtype=float)
    for i in range(0, len(y)):
        x_Mat[i, :] = x
    y_Mat = np.zeros((len(y), len(x)), dtype=float)
    for i in range(0, len(x)):
        y_Mat[:, i] = y
    epsilon1 = (-P / (E * I)) * np.multiply(x_Mat, y_Mat)
    epsilon2 = ((nu * P) / (E * I)) * np.multiply(x_Mat, y_Mat)
    epsilon6 = (P / (2 * I * G)) * (y_Mat ** 2 - c ** 2)
    return [epsilon1, epsilon2, epsilon6]


def dic_epsilon(dataFile):
    data = scio.loadmat(dataFile)
    cur_5 = data['strains'][0][4]   # 提取第5张图的数据[4]代表第五张图的数据
    epsilon1, epsilon2, epsilon6 = cur_5[0], cur_5[1], cur_5[2]
    return [epsilon1, epsilon2, epsilon6]