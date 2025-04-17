from pathlib import Path
from OpenGL.GL import *

from .vertices import create_cube_vertices


def setup_vao_vbo():
    vertices = create_cube_vertices()

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 位置属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # 颜色属性
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, len(vertices) // 6  # 返回VAO和顶点数量
