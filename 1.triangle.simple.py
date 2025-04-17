import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# 最简单的顶点着色器
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 color;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    color = aColor;
}
"""

# 最简单的片段着色器（固定红色）
FRAGMENT_SHADER = """
#version 330 core
in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // 纯红色
    FragColor = vec4(color, 1.0);
}
"""


def main():
    # 初始化GLFW
    if not glfw.init():
        return

    # 创建窗口
    window = glfw.create_window(800, 600, "Red Triangle Test", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 定义三角形顶点数据 (NDC坐标，范围[-1,1])
    vertices = np.array([
        # x, y, z
        -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,  # 左下
        0.5, -0.5, 0.0, 0.0, 1.0, 0.0,  # 右下
        0.0,  0.5, 0.0, 0.0, 0.0, 1.0   # 顶部
    ], dtype=np.float32)

    # 创建VAO和VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 设置顶点属性指针
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # 设置颜色属性指针
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          6 * 4, ctypes.c_void_p(3*4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # 编译着色器
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # 渲染循环
    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)  # 背景色
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader)
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)  # 绘制3个顶点

        glfw.swap_buffers(window)
        glfw.poll_events()

    # 清理资源
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader)

    glfw.terminate()


if __name__ == "__main__":
    main()
