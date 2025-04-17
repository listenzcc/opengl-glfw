import glm
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from cube.vao_vbo import setup_vao_vbo

FRAGMENT_SHADER = open('./cube/shader.frag').read()
VERTEX_SHADER = open('./cube/shader.vert').read()


def key_callback(window, key, scancode, action, mods):
    '''Keyboard event callback'''
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        print("ESC is pressed, bye bye.")
        # db.save()
        glfw.set_window_should_close(window, True)
        return
    elif action == glfw.PRESS:
        print(f"Key pressed {key}")
    elif action == glfw.RELEASE:
        # print(f"Key released {key}")
        return
    elif action == glfw.REPEAT:
        # print(f"Key repeated {key}")
        return
    return


def main():
    if not glfw.init():
        raise RuntimeError('Failed initialize GLFW.')

    window = glfw.create_window(
        800, 600, "Window", monitor=None, share=None)
    if not window:
        glfw.terminate()
        raise RuntimeError('Failed initialize window.')

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)

    VAO, vertex_count = setup_vao_vbo()

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # 获取uniform位置
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    projection_loc = glGetUniformLocation(shader, "projection")

    while not glfw.window_should_close(window):
        glClearColor(0.0, 0.2, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
        # 渲染立方体
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)

        glfw.swap_buffers(window)
        glfw.poll_events()


if __name__ == '__main__':
    main()
