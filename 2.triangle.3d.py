import glm
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

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
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


def perspective(fovy, aspect, near, far):
    return glm.perspective(glm.radians(fovy), aspect, near, far)


def lookAt(eye, center, up):
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return view


def rotate(matrix, angle, axis):
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c

    rot = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s, 0],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s, 0],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c,   0],
        [0,           0,           0,           1]
    ], dtype=np.float32)

    return np.dot(matrix, rot)


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

    # 获取uniform位置
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    projection_loc = glGetUniformLocation(shader, "projection")

    # 渲染循环
    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)  # 背景色
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader)

        # 使用GLM生成矩阵
        model = glm.mat4(1.0)
        model = glm.rotate(model, glfw.get_time(), glm.vec3(0.5, 1.0, 0.0))

        view = glm.lookAt(
            glm.vec3(2, 2, 2),  # 相机位置
            glm.vec3(0, 0, 0),   # 观察目标
            glm.vec3(0, 1, 0)    # 上向量
        )

        projection = glm.perspective(
            glm.radians(45.0),   # 视野角度
            800/600,             # 宽高比
            0.1,                 # 近平面
            100.0                # 远平面
        )

        # 传递矩阵到着色器
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE,
                           glm.value_ptr(projection))
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
