import numpy as np


def create_cube_vertices():
    # 顶点格式: [x, y, z, r, g, b]
    vertices = np.array([
        # 前面 (红色)
        -0.5, -0.5,  0.5, 1.0, 0.0, 0.0,
        0.5, -0.5,  0.5, 1.0, 0.0, 0.0,
        0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
        0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
        -0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
        -0.5, -0.5,  0.5, 1.0, 0.0, 0.0,

        # 后面 (绿色)
        -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
        0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
        0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
        0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
        -0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
        -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,

        # 左面 (蓝色)
        -0.5,  0.5,  0.5, 0.0, 0.0, 1.0,
        -0.5,  0.5, -0.5, 0.0, 0.0, 1.0,
        -0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
        -0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
        -0.5, -0.5,  0.5, 0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5, 0.0, 0.0, 1.0,

        # 右面 (黄色)
        0.5,  0.5,  0.5, 1.0, 1.0, 0.0,
        0.5,  0.5, -0.5, 1.0, 1.0, 0.0,
        0.5, -0.5, -0.5, 1.0, 1.0, 0.0,
        0.5, -0.5, -0.5, 1.0, 1.0, 0.0,
        0.5, -0.5,  0.5, 1.0, 1.0, 0.0,
        0.5,  0.5,  0.5, 1.0, 1.0, 0.0,

        # 上面 (青色)
        -0.5,  0.5, -0.5, 0.0, 1.0, 1.0,
        0.5,  0.5, -0.5, 0.0, 1.0, 1.0,
        0.5,  0.5,  0.5, 0.0, 1.0, 1.0,
        0.5,  0.5,  0.5, 0.0, 1.0, 1.0,
        -0.5,  0.5,  0.5, 0.0, 1.0, 1.0,
        -0.5,  0.5, -0.5, 0.0, 1.0, 1.0,

        # 下面 (品红)
        -0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
        0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
        0.5, -0.5,  0.5, 1.0, 0.0, 1.0,
        0.5, -0.5,  0.5, 1.0, 0.0, 1.0,
        -0.5, -0.5,  0.5, 1.0, 0.0, 1.0,
        -0.5, -0.5, -0.5, 1.0, 0.0, 1.0
    ], dtype=np.float32)

    # vertices = np.random.random(vertices.shape)

    return vertices
