import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# 初始化 GLFW
if not glfw.init():
    raise Exception("GLFW 初始化失败")

# 创建窗口
window = glfw.create_window(800, 600, "OpenGL 光线追踪", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW 窗口创建失败")

glfw.make_context_current(window)

# 顶点着色器 (简单传递)
vertex_shader = """
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
}
"""

# 片段着色器 (包含光线追踪逻辑)
fragment_shader = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform float iTime;
uniform vec2 iResolution;

// 光线追踪函数
struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;
};

struct HitRecord {
    float t;
    vec3 point;
    vec3 normal;
    vec3 color;
    float reflectivity;
};

// 场景中的球体
Sphere spheres[3];

// 初始化场景
void initScene() {
    spheres[0] = Sphere(vec3(0.0, -100.5, -1.0), 100.0, vec3(0.8, 0.8, 0.0), 0.2);
    spheres[1] = Sphere(vec3(0.0, 0.0, -1.0), 0.5, vec3(0.7, 0.3, 0.3), 0.5);
    spheres[2] = Sphere(vec3(1.0, 0.0, -1.0), 0.5, vec3(0.8, 0.6, 0.2), 0.8);
}

// 球体相交检测
bool hitSphere(Sphere sphere, Ray ray, float t_min, float t_max, out HitRecord rec) {
    vec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - a * c;
    
    if (discriminant > 0.0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = ray.origin + rec.t * ray.direction;
            rec.normal = (rec.point - sphere.center) / sphere.radius;
            rec.color = sphere.color;
            rec.reflectivity = sphere.reflectivity;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = ray.origin + rec.t * ray.direction;
            rec.normal = (rec.point - sphere.center) / sphere.radius;
            rec.color = sphere.color;
            rec.reflectivity = sphere.reflectivity;
            return true;
        }
    }
    return false;
}

// 场景相交检测
bool hitWorld(Ray ray, float t_min, float t_max, out HitRecord rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    for (int i = 0; i < 3; i++) {
        if (hitSphere(spheres[i], ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

// 计算光线颜色
vec3 rayColor(Ray ray) {
    HitRecord rec;
    vec3 color = vec3(1.0);
    
    for (int i = 0; i < 10; i++) {
        if (hitWorld(ray, 0.001, 1000.0, rec)) {
            vec3 target = rec.point + rec.normal + vec3(
                sin(iTime * 2.0) * 0.5,
                cos(iTime * 1.5) * 0.5,
                sin(iTime * 1.0) * 0.5
            );
            ray = Ray(rec.point, normalize(target - rec.point));
            color *= rec.color * rec.reflectivity;
        } else {
            // 背景色
            vec3 unit_direction = normalize(ray.direction);
            float t = 0.5 * (unit_direction.y + 1.0);
            color *= mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
            break;
        }
    }
    
    return color;
}

void main()
{
    initScene();
    
    // 设置虚拟相机
    vec2 uv = (2.0 * gl_FragCoord.xy - iResolution.xy) / iResolution.y;
    
    Ray ray;
    ray.origin = vec3(sin(iTime * 0.5) * 2.0, cos(iTime * 0.3), 1.0);
    ray.direction = normalize(vec3(uv, -1.0));
    
    vec3 color = rayColor(ray);
    
    // Gamma校正
    color = sqrt(color);
    
    FragColor = vec4(color, 1.0);
}
"""

# 编译着色器
compileShader(vertex_shader, GL_VERTEX_SHADER)
compileShader(fragment_shader, GL_FRAGMENT_SHADER)
try:
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
except Exception as e:
    print("Shader compilation failed:")
    print(e)
    # 打印顶点着色器错误
    try:
        compileShader(vertex_shader, GL_VERTEX_SHADER)
    except Exception as ve:
        print("\nVertex shader error:")
        print(ve)
    # 打印片段着色器错误
    try:
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    except Exception as fe:
        print("\nFragment shader error:")
        print(fe)
    glfw.terminate()
    exit(1)

# 定义全屏四边形
vertices = np.array([
    # 位置       # 纹理坐标
    -1.0, -1.0,  0.0, 0.0,
    1.0, -1.0,  1.0, 0.0,
    -1.0,  1.0,  0.0, 1.0,
    1.0,  1.0,  1.0, 1.0
], dtype=np.float32)

# 创建VAO和VBO
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# 位置属性
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# 纹理坐标属性
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
glEnableVertexAttribArray(1)

# 主循环
glUseProgram(shader)
resolution_loc = glGetUniformLocation(shader, "iResolution")
time_loc = glGetUniformLocation(shader, "iTime")

while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT)

    # 更新uniform变量
    width, height = glfw.get_window_size(window)
    glUniform2f(resolution_loc, width, height)
    glUniform1f(time_loc, glfw.get_time())

    # 渲染全屏四边形
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    glfw.swap_buffers(window)
    glfw.poll_events()

# 清理资源
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteProgram(shader)
glfw.terminate()
