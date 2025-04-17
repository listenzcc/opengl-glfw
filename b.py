import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# 顶点着色器
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
"""

# 片段着色器
FRAGMENT_SHADER = """
#version 330 core
in vec3 ourColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
}
"""


def create_cube_vertices():
    # 位置 + 颜色 (每个面不同颜色)
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
    return vertices


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

    return VAO, VBO, len(vertices) // 6


def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (zfar + znear) / (znear - zfar)
    projection[2, 3] = (2 * zfar * znear) / (znear - zfar)
    projection[3, 2] = -1.0
    return projection


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
    if not glfw.init():
        raise RuntimeError('Failed to initialize GLFW')

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Rotating Cube", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('Failed to create window')

    glfw.make_context_current(window)
    glfw.set_key_callback(window, lambda win, key, sc, action, mods:
                          glfw.set_window_should_close(win, True) if key == glfw.KEY_ESCAPE else None)

    glEnable(GL_DEPTH_TEST)

    VAO, VBO, vertex_count = setup_vao_vbo()

    # 编译着色器
    def compile_shader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compilation error:\n{error}")
        return shader

    vertex_shader = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

    shader = glCreateProgram()
    glAttachShader(shader, vertex_shader)
    glAttachShader(shader, fragment_shader)
    glLinkProgram(shader)

    if not glGetProgramiv(shader, GL_LINK_STATUS):
        error = glGetProgramInfoLog(shader).decode()
        raise RuntimeError(f"Program linking error:\n{error}")

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    # 获取uniform位置
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    projection_loc = glGetUniformLocation(shader, "projection")

    if -1 in (model_loc, view_loc, projection_loc):
        raise RuntimeError("Failed to get uniform locations")

    # 渲染循环
    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # 模型矩阵 (旋转)
        model = np.identity(4, dtype=np.float32)
        model = rotate(model, glfw.get_time() * 50, [0.5, 1.0, 0.0])

        # 视图矩阵 (相机)
        view = lookAt(
            eye=np.array([2.0, 2.0, 2.0]),
            center=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0])
        )

        # 投影矩阵
        projection = perspective(45.0, 800/600, 0.1, 100.0)

        # 传递矩阵
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

        # 渲染
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader)
    glfw.terminate()


if __name__ == "__main__":
    main()
