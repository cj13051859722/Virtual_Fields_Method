import numpy as np
import random
from scipy.optimize import minimize, fmin_cobyla
# 自建函数
from epsilon.epsilon import real_epsilon_fig2_7, dic_epsilon


def residual(input_prop):
    """
    平面应力问题
    """
    # step1 定义
    # step1.1 尺寸
    c = l / 2
    mesh_size = Step * mmPerPix
    x = np.arange(mesh_size / 2, L, mesh_size)
    y = np.arange(c - mesh_size / 2, -c, -mesh_size)
    SmallArea = (mesh_size) ** 2  # in mm^2
    x_Mat = np.zeros((len(y), len(x)), dtype=float)
    for i in range(0, len(y)):
        x_Mat[i, :] = x
    y_Mat = np.zeros((len(y), len(x)), dtype=float)
    for i in range(0, len(x)):
        y_Mat[:, i] = y
    # step1.2 弹模、泊松比
    input_E = input_prop[0]
    input_nu = input_prop[1]
    Q11 = input_E / (1 - input_nu ** 2)
    Q12 = input_nu * Q11
    Q66 = (Q11 - Q12) / 2.

    # step2 简化公式3.4，将实际应变合并为暂时应变
    sigma1_tmp = Q11 * epsilon1 + Q12 * epsilon2
    sigma2_tmp = Q12 * epsilon1 + Q11 * epsilon2
    sigma6_tmp = Q66 * epsilon6

    # step3 设置各虚场

    # step3.1 Linear Virtual Extension
    # u_Mat_vf1 = x_Mat - L
    # v_Mat_vf1 = np.zeros((len(y),len(x)),dtype=float)

    # Virtual strains #1 in MPa
    # epsilon1_vf1 = np.ones((len(y),len(x)),dtype=float)
    # epsilon2_vf1 = np.zeros((len(y),len(x)),dtype=float)
    # epsilon6_vf1 = np.zeros((len(y),len(x)),dtype=float)

    # Internal virtual work #1 (J)
    # Wint_vf1 = - np.sum((np.multiply(sigma1_tmp,epsilon1_vf1) + np.multiply(sigma2_tmp,epsilon2_vf1) + np.multiply(sigma6_tmp,epsilon6_vf1)) * SmallArea)

    # External virtual work #1 (J)
    # Wext_vf1 = 0 because force and displacement are orthogonal
    # Wext_vf1 = P * 0.

    # step3.2 Vertical Linear Virtual Displacement
    # u_Mat_vf2 = np.zeros((len(y),len(x)),dtype=float)
    # v_Mat_vf2 = x_Mat - L
    # Internal virtual work
    epsilon1_vf2 = np.zeros((len(y), len(x)), dtype=float)  # TODO 更改虚场
    epsilon2_vf2 = np.zeros((len(y), len(x)), dtype=float)
    epsilon6_vf2 = np.ones((len(y), len(x)), dtype=float)
    Wint_vf2 = - np.sum((np.multiply(sigma1_tmp, epsilon1_vf2) + np.multiply(sigma2_tmp, epsilon2_vf2)
                         + np.multiply(sigma6_tmp, epsilon6_vf2)) * SmallArea)
    # External virtual work
    # Wext_vf2 = P * (x - L)  with x = 0
    Wext_vf2 = P * (-L)

    # step3.3 Parabolic Virtual Deflection
    # u_Mat_vf3 = - np.multiply((x_Mat-L),y_Mat)
    # v_Mat_vf3 = 0.5 * (x_Mat - L)**2

    # Internal virtual work
    epsilon1_vf3 = -y_Mat
    epsilon2_vf3 = np.zeros((len(y), len(x)), dtype=float)
    epsilon6_vf3 = np.zeros((len(y), len(x)), dtype=float)
    Wint_vf3 = - np.sum((np.multiply(sigma1_tmp, epsilon1_vf3) + np.multiply(sigma2_tmp, epsilon2_vf3) + np.multiply(
        sigma6_tmp, epsilon6_vf3)) * SmallArea)
    # External virtual work
    # Wext_vf3 = P * (x - L)**2 / 2  with  x = 0
    Wext_vf3 = P * 0.5 * (-L) ** 2

    # step3.4 Sinusoidal Virtual Displacement
    # u_Mat_vf4 = np.multiply(y_Mat**3, np.sin(2. * np.pi * (x_Mat-L) / L))
    # v_Mat_vf4 = np.zeros((len(y),len(x)),dtype=float)

    # Virtual strains #4 in MPa
    # epsilon1_vf4 = (2. * np.pi / L) * np.multiply(y_Mat**3, np.cos(2. * np.pi * (x_Mat-L) / L))
    # epsilon2_vf4 = np.zeros((len(y),len(x)),dtype=float)
    # epsilon6_vf4 = 3.* np.multiply(y_Mat**2, np.sin(2. * np.pi * (x_Mat-L) / L))

    # Internal virtual work #4 (J)
    # Wint_vf4 = - np.sum((np.multiply(sigma1_tmp,epsilon1_vf4) + np.multiply(sigma2_tmp,epsilon2_vf4) + np.multiply(sigma6_tmp,epsilon6_vf4)) * SmallArea)

    # External virtual work #4 (J)
    # Wext_vf4 = 0 because force and displacement are orthogonal
    # Wext_vf4 = P * 0.

    #####

    # Residual function
    # res = np.sqrt(((Wint_vf1 + Wext_vf1)**2 + (Wint_vf2 + Wext_vf2)**2 + (Wint_vf3 + Wext_vf3)**2 + (Wint_vf4 + Wext_vf4)**2) / (Wext_vf1**2 + Wext_vf2**2 + Wext_vf3**2 + Wext_vf4**2))
    res = np.sqrt(((Wint_vf2 + Wext_vf2) ** 2 + (Wint_vf3 + Wext_vf3) ** 2) / (Wext_vf2 ** 2 + Wext_vf3 ** 2))

    # print "Wint_vf1 =", Wint_vf1
    # print "Wext_vf1 =", Wext_vf1
    # print "Wint_vf2 =", Wint_vf2
    # print "Wext_vf2 =", Wext_vf2
    # print "Wint_vf3 =", Wint_vf3
    # print "Wext_vf3 =", Wext_vf3
    # print "Wint_vf4 =", Wint_vf4
    # print "Wext_vf4 =", Wext_vf4

    # print "Residual =", res
    return res


if __name__ == "__main__":
    # step 1各参数
    # step 1.1尺寸
    L = 75.  # Length mm
    l = 25.  # Width
    Step = 20.  # in pixels => change this value to change the mesh size
    mmPerPix = 0.05  # mm / pixel ratio
    # step 1.2载荷大小
    P = -50.
    # step 1.3实际应变场大小
    E = 4000.
    nu = 0.3
    t = 1
    epsilon1, epsilon2, epsilon6 = real_epsilon_fig2_7(P=P, E=E, nu=nu, L=L, l=l, t=t, Step=Step,
                                                mmPerPix=mmPerPix)  # 模拟值
    # epsilon1, epsilon2, epsilon6 = dic_epsilon()  # TODO DIC模拟值

    # step 2值估算
    input_prop = np.array((random.uniform(0.02, 20.) * 500., random.uniform(0., 2.) * 0.25), dtype=float)
    output_prop = minimize(residual, input_prop, method='L-BFGS-B', bounds=((10., 10000.), (0., 0.5)))

    # step3 对比
    E = 4000.
    nu = 0.3
    print("E =", E)
    print("E_vfm =", output_prop.x[0])
    print("Error_rel_E", (output_prop.x[0] - E) / E * 100., "%")
    print("nu =", nu)
    print("nu_vfm =", output_prop.x[1])
    print("Error_rel_nu", (output_prop.x[1] - nu) / nu * 100., "%")
