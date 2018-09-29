import numpy as np

from OpenGL.GL import *
from OpenGL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import glfw

import os
import sys
import cv2
import struct

import StringIO
from plyfile import PlyData

CUR_MODULE_DIR, _ = os.path.split(__file__)
SHADER_DIR = os.path.join(CUR_MODULE_DIR, "shaders")
PLY_DIR = os.path.join(CUR_MODULE_DIR, "plymodels")

class PathError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value + ': is not a file or not readable!'

def error_callback(error, desc):
    print("glfw Error(%d): %s" % (error, desc))

class MyApplication():
    def __init__(self, title="Application", wndWidth=368, wndHeight=368, is_visible=True):
        self.title = title
        self.wndHeight = wndHeight
        self.wndWidth = wndWidth
        self.window = None
        self.is_visible = is_visible

    def glfwInit(self, button_callback = None, mouse_down_callback = None, mouse_move_callback = None):

        glfw.set_error_callback(error_callback)

        if not glfw.init():
            print("glfw Error: init glfw failed!")
            return False
        glfw.window_hint(glfw.RESIZABLE, GL_FALSE)
        glfw.window_hint(glfw.SAMPLES, 1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if self.is_visible:
            glfw.window_hint(glfw.VISIBLE, GL_TRUE)
        else:
            glfw.window_hint(glfw.VISIBLE, GL_FALSE)

        self.window = glfw.create_window(self.wndWidth, self.wndHeight, self.title, None, None)
        if not self.window:
            print("glfw Error: create window failed!")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, GL_TRUE)

        if button_callback is not None and hasattr(button_callback, "__call__"):
            glfw.set_key_callback(self.window, button_callback)
        if mouse_down_callback is not None and hasattr(mouse_down_callback, "__call__"):
            glfw.set_mouse_button_callback(self.window, mouse_down_callback)
        if mouse_move_callback is not None and hasattr(mouse_move_callback, "__call__"):
            glfw.set_cursor_pos_callback(self.window, mouse_move_callback)

        # Enable the Depth test
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDepthFunc(GL_LESS)
        # glEnable(GL_LINE_SMOOTH)
        return True
    def windowShouldClose(self):
        return glfw.window_should_close(self.window)
    def setWindowShouldClose(self, close):
        glfw.set_window_should_close(self.window, close)

    def renderStart(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def renderEnd(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def captureFrame(self):
        # call after swap_buffers
        glReadBuffer(GL_FRONT)
        frame = glReadPixels(0, 0, self.wndWidth, self.wndHeight, GL_RGB, GL_UNSIGNED_BYTE)

        frame = np.reshape(struct.unpack("%dB" % (self.wndWidth * self.wndHeight * 3), frame), [self.wndHeight, self.wndWidth, 3]).astype(np.uint8)

        frame = cv2.flip(frame, 0)[:, :, ::-1]

        return frame
    def terminate(self):
        glfw.terminate()

class OpenGLUtils():
    def __init__(self):
        pass

    @staticmethod
    def perspective(fov, ratio, near, far, is_transpose = False):
        matrix = np.zeros((4, 4))
        matrix[1][1] = 1.0 / np.tan(fov/2.0)
        matrix[0][0] = matrix[1][1] / ratio
        matrix[3][2] = -1
        matrix[2][2] = -(far + near) / (far - near)
        matrix[2][3] = -2 * far * near / (far - near)

        if is_transpose:
            return np.transpose(matrix, [1, 0])
        else:
            return matrix
    @staticmethod
    def ortho(left, right, bottom, top, near, far, is_transpose=False):
        tx = -float(right + left) / (right - left)
        ty = -float(top + bottom) / (top - bottom)
        tz = -float(far + near) / (far - near)

        mat = np.array([
            [2.0/(right - left), 0, 0, tx],
            [0, 2.0/(top - bottom), 0, ty],
            [0, 0, -2.0/(far - near), tz],
            [0, 0, 0, 1]
            ])

        if is_transpose:
            return np.transpose(mat, [1, 0])
        else:
            return mat
    # input numpy.array
    @classmethod
    def normalize(cls, data):
        data_sum = pow(sum([i*i for i in data]), 0.5) + 0.0000001
        result = np.array(data) / data_sum
        return result

    @classmethod
    def lookAt(cls, eye, target, up, is_transpose=False):
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)
        # TODO I can't understand why it's that
        Z = cls.normalize(eye - target)
        X = cls.normalize(np.cross(up, Z))
        Y = np.cross(Z, X)

        mat = np.array([
            [X[0], Y[0], Z[0], 0.0],
            [X[1], Y[1], Z[1], 0.0],
            [X[2], Y[2], Z[2], 0.0],
            [np.dot(X, -eye), np.dot(Y, -eye), np.dot(Z, -eye), 1.0]
            ])
        if not is_transpose:
            return np.transpose(mat, [1,0])
        else:
            return mat
    @classmethod
    def translate(cls, target, is_transpose=False):
        mat = np.array([
            [1, 0, 0, target[0]],
            [0, 1, 0, target[1]],
            [0, 0, 1, target[2]],
            [0, 0, 0, 1]
            ], dtype=np.float32)
        if is_transpose:
            return np.transpose(mat, [1, 0])
        else:
            return mat

    @classmethod
    def rotate(cls, axis, theta, is_transpose=False):
        axis = cls.normalize(axis)
        u = axis[0]
        v = axis[1]
        w = axis[2]

        mat = np.zeros((4, 4), dtype=np.float32)

        mat[0][0] = np.cos(theta) + (u * u) * (1 - np.cos(theta));
        mat[0][1] = u * v * (1 - np.cos(theta)) + w * np.sin(theta);
        mat[0][2] = u * w * (1 - np.cos(theta)) - v * np.sin(theta);
        mat[0][3] = 0;

        mat[1][0] = u * v * (1 - np.cos(theta)) - w * np.sin(theta);
        mat[1][1] = np.cos(theta) + v * v * (1 - np.cos(theta));
        mat[1][2] = w * v * (1 - np.cos(theta)) + u * np.sin(theta);
        mat[1][3] = 0;

        mat[2][0] = u * w * (1 - np.cos(theta)) + v * np.sin(theta);
        mat[2][1] = v * w * (1 - np.cos(theta)) - u * np.sin(theta);
        mat[2][2] = np.cos(theta) + w * w * (1 - np.cos(theta));
        mat[2][3] = 0;

        mat[3][0] = 0;
        mat[3][1] = 0;
        mat[3][2] = 0;
        mat[3][3] = 1;

        if is_transpose:
            return mat
        else:
            return np.transpose(mat, [1, 0])
    @classmethod
    def scale(cls, ratio):
        return np.array([
            [ratio[0], 0, 0, 0],
            [0, ratio[1], 0, 0],
            [0, 0, ratio[2], 0],
            [0, 0, 0, 1]
            ])

class ShaderReader():
    def __init__(self, v_path = None, f_path = None):
        self.program = glCreateProgram()

        if not v_path or not f_path or not os.path.exists(v_path) or not os.path.exists(f_path):
            print("Error: The vertex shader path or the fragment shader path is not valid!")
            raise PathError("Shader file")

        with open(v_path) as v_file:
            vertexShaderCode = v_file.read()
        with open(f_path) as f_file:
            fragShaderCode = f_file.read()
        self.vertexShader = self.create_shader(vertexShaderCode, GL_VERTEX_SHADER)
        self.fragShader = self.create_shader(fragShaderCode, GL_FRAGMENT_SHADER)

        glAttachShader(self.program, self.vertexShader)
        glAttachShader(self.program, self.fragShader)

        glLinkProgram(self.program)
        message = self.get_program_log(self.program)
        if message:
            print("Shader: shader program message: %s" % message)

    def create_shader(self, source, shadertype):
        shader = glCreateShader(shadertype)
        if isinstance(source, basestring):
            source = [source]

        glShaderSource(shader, source)
        glCompileShader(shader)
        message = self.get_shader_log(shader)
        if message:
            print("Shader: shader message: %s" % message)

        return shader

    def get_shader_log(self, shader):
        return self.get_log(shader, glGetShaderInfoLog)

    def get_program_log(self, shader):
        return self.get_log(shader, glGetProgramInfoLog)

    def get_log(self, obj, func):
        value = func(obj)
        return value

    def use(self):
        glUseProgram(self.program)

    def stop(self):
        glUseProgram(0)

    def set_uniform_f(self, name, value):
        location = glGetUniformLocation(self.program, name)
        glUniform1f(location, value)

    def set_uniform_i(self, name, value):
        location = glGetUniformLocation(self.program, name)
        glUniform1i(location, value)

    def set_uniform_ui(self, name, value):
        location = glGetUniformLocation(self.program, name)
        glUniform1ui()

    def set_uniform_m4(self, name, value):
        location = glGetUniformLocation(self.program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, value)

    def set_uniform_v3(self, name, value):
        location = glGetUniformLocation(self.program, name)
        glUniform3fv(location, 1, value)

    def __setitem__(self, name, value):
        if len(name) == 2:
            if name[1] == (1, 3):
                self.set_uniform_v3(name[0], value)
        else:
            if isinstance(value, float) or isinstance(value, np.float32):
                # print("Set uniform float data!")
                self.set_uniform_f(name, value)
            elif isinstance(value, int):
                # print("Set uniform int data!")
                self.set_uniform_i(name, value)
            elif len(value) == 3:
                # print("Set uniform mat3 data!")
                self.set_uniform_v3(name, value)
            elif len(value) == 4:
            # print("Set uniform mat4 data!")
                self.set_uniform_m4(name, value)

class MeshEntry():
    def __init__(self, vertexs, faces, norms, vao):
        self.VAO = vao
        self.elmNum = len(faces)
        self.n_face_indices = faces.shape[1]
        glBindVertexArray(self.VAO)

        # put in the vertexs
        self.vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertexs.flatten(), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # put in the indices
        self.indices_buffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(faces, dtype=np.uint32).flatten(), GL_STATIC_DRAW)

        # put in the norms
        self.norm_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)
        glBufferData(GL_ARRAY_BUFFER, norms.flatten(), GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def render(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)

        glDrawElements(GL_TRIANGLES, self.n_face_indices * self.elmNum, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

class MeshRenderer():
    def __init__(self):
        self.VAO = glGenVertexArrays(1)
        self.meshes = []

    # mesh_file is the path or the readable object
    def addMesh(self, mesh_file):
        if not hasattr(mesh_file, "read") and not os.path.isfile(mesh_file):
            raise PathError(mesh_file)

        plydata = PlyData.read(mesh_file)

        mesh_vertices = np.concatenate([plydata["vertex"]['x'][:, np.newaxis], plydata["vertex"]['y'][:, np.newaxis], plydata["vertex"]['z'][:, np.newaxis]], axis=1)
        mesh_norms = np.concatenate([plydata["vertex"]['nx'][:, np.newaxis], plydata["vertex"]['ny'][:, np.newaxis], plydata["vertex"]['nz'][:, np.newaxis]], axis=1)
        mesh_faces = []

        for face in plydata["face"].data:
            face = face[0]
            if len(face) > 3:
                for i in range(len(face) - 2):
                    tmp_face = [face[0], face[i+1], face[i+2]]
                    mesh_faces.append(tmp_face)
            else:
                mesh_faces.append(face)
        self.meshes.append(MeshEntry(mesh_vertices, np.array(mesh_faces), mesh_norms, self.VAO))

    def render(self, mesh_num=0):
        self.meshes[mesh_num].render()


class PoseModel():
    def __init__(self, mesh_render, shader, cam_win_size, proj_mat, view_mat, limbs_n_root = []):
        self.shader = shader
        self.mesh_render = mesh_render
        if len(limbs_n_root) == 0:
            self.joints_root = 14
            self.limbs = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 14],
                [8, 14],
                [8, 9],
                [9, 10],
                [14, 11],
                [11, 12],
                [12, 13]
            ], dtype=np.uint8)
        else:
            self.limbs = limbs_n_root[0]
            self.joints_root = limbs_n_root[1]

        self.rotateMat = np.identity(4)
        self.prev_rotateMat = np.identity(4)
        self.cam_img_width = cam_win_size[0]
        self.cam_img_height = cam_win_size[1]

        self.Proj = proj_mat
        self.View = view_mat

        # write the calculateed project&clip matrix
        # for the real camera the far is infinity and near is f
        # if is_ar:
            # self.Proj = np.array([
                # [2.0*cam_matrix[0][0] / self.cam_img_width, 0, -1 + 2.0*cam_matrix[0][2] / self.cam_img_width, 0.0],
                # [0, -2.0*cam_matrix[1][1]/self.cam_img_height, 1 - 2.0*cam_matrix[1][2] / self.cam_img_height, 0.0],
                # [0, 0, 1, -2 * cam_matrix[0][0]],
                # [0, 0, 1, 0]
                # ], dtype=np.float32)
        # else:
            # self.Proj = cam_matrix
    def setProjMat(self, proj_mat):
        self.Proj = proj_mat

    def setViewMat(self, view_mat):
        self.View = view_mat

    def rotate(self, rotateMat=np.identity(4), reset=False, set_prev=False):
        global is_mouse_pressed
        if reset:
            self.rotateMat = np.identity(4)
        else:
            if not set_prev:
                self.rotateMat = np.dot(rotateMat, self.prev_rotateMat)
            else:
                self.prev_rotateMat = self.rotateMat

    def draw(self, vertexs, color=[1.0, 0.8, 1.0], sphere_ratio=0.04):
        # I get that, in the vertex shader OpenGL don't divide the w
        vertexs = vertexs.copy()
        origin_p = vertexs[self.joints_root].copy()
        for i in range(vertexs.shape[0]):
            vertexs[i] -= origin_p
            vertexs[i] = np.dot(self.rotateMat, np.concatenate((vertexs[i], [1.0])))[0:3]
            vertexs[i] += origin_p

        vertex_num = vertexs.shape[0]
        line_num = self.limbs.shape[0]

        vertex_flags = [0] * vertex_num
        self.shader.use()
        # use the realword unit
        self.shader["viewPos", (1, 3)] = np.array([0, 0, 10])
        self.shader["lightPos", (1, 3)] = np.array([0, 0, 100])
        self.shader["fragColor", (1, 3)] = np.array(color)

        for i in range(line_num):
            line = [self.limbs[i][0], self.limbs[i][1]]
            model_mat = np.identity(4)

            for j in range(2):
                if not vertex_flags[line[j]]:
                    vertex_flags[line[j]] = 1

                    trans_mat = OpenGLUtils.translate(vertexs[line[j]])
                    scale_mat = OpenGLUtils.scale([1, 1, 1])

                    cur_model_mat = np.dot(trans_mat, scale_mat)
                    model_mat = np.transpose(cur_model_mat, [1, 0])

                    self.shader["MVP"] = np.transpose(np.dot(self.Proj, np.dot(self.View, cur_model_mat)))
                    self.shader["modelMat"] = model_mat
                    self.shader["normMat"] = np.transpose(np.linalg.inv(model_mat))
                    self.mesh_render.render(0)
                    # render the dot mesh
            point_a = vertexs[line[0]]
            point_b = vertexs[line[1]]

            line_center = (point_a + point_b) / 2.0
            length = np.linalg.norm(point_a - point_b)
            # then render the line mesh
            vFrom = np.array([0, 0, 1])
            vTo = OpenGLUtils.normalize(point_a - point_b)

            trans_mat = OpenGLUtils.translate(line_center)
            angle = np.arccos(np.dot(vFrom, vTo))
            if angle <= 0.000001:
                rotate_mat = np.identity(4)
            else:
                rotate_mat = OpenGLUtils.rotate(OpenGLUtils.normalize(np.cross(vFrom, vTo)), angle)

            scale_mat = OpenGLUtils.scale([1, 1, length / (2.0 * sphere_ratio)])
            curmodel = np.dot(trans_mat, np.dot(rotate_mat, scale_mat))
            tran_curmodel = np.transpose(curmodel, [1, 0])
            self.shader["MVP"] = np.transpose(np.dot(self.Proj, np.dot(self.View, curmodel)))
            self.shader["modelMat"] = tran_curmodel
            self.shader["normMat"] = np.transpose(np.linalg.inv(tran_curmodel))
            self.mesh_render.render(0)

class CamScene():
    def __init__(self, cam_shader, wndWidth, wndHeight):
        self.wndWidth = wndWidth
        self.wndHeight = wndHeight
        self.cam_shader = cam_shader
        self.base_fov = 45.0
        self.VAO = glGenVertexArrays(1)
        self.z_pos = 99.0
        self.proj_mat = None

        glBindVertexArray(self.VAO)
        self.initGLFrame()
        glBindVertexArray(0)

    def initGLFrame(self):
        projMat = OpenGLUtils.perspective(np.radians(self.base_fov), float(self.wndWidth) / self.wndHeight, 0.1, 100.0)
        self.proj_mat = projMat
        modelMat = np.identity(4)
        self.MVP = np.transpose(np.dot(projMat, modelMat))
        self.genTexture()
        tmp_t = self.z_pos * np.tan(np.radians(self.base_fov/2))
        tmp_r = tmp_t * self.wndWidth / self.wndHeight
        vertex_buffer_data = np.array([
            -tmp_r, tmp_t, -self.z_pos,
            -tmp_r, -tmp_t, -self.z_pos,
            tmp_r, -tmp_t, -self.z_pos,
            tmp_r, -tmp_t, -self.z_pos,
            tmp_r, tmp_t, -self.z_pos,
            -tmp_r, tmp_t, -self.z_pos], dtype=np.float32)
        uv_data = np.array([
                1.0, 1.0,
                1.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 1.0,
                1.0, 1.0], dtype=np.float32)

        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, vertex_buffer_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        self.uvBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.uvBuffer)
        glBufferData(GL_ARRAY_BUFFER, uv_data, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))


    def genTexture(self):
        self.textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    def setTextureData(self, frame):
        frame = cv2.flip(frame, -1)
        frame = frame.tobytes()
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.wndWidth, self.wndHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)
        glGenerateMipmap(GL_TEXTURE_2D)

    def drawFrame(self, frame):
        glBindVertexArray(self.VAO)
        self.cam_shader.use()
        self.cam_shader["MVP"] = self.MVP

        frame = cv2.resize(frame, (self.wndWidth, self.wndHeight))
        self.setTextureData(frame)

        # The default activated texture is 0, so this line is not necessary
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        self.cam_shader["myTextureSampler"] = 0

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glBindVertexArray(0)

global init_x, init_y, cur_x, cur_y, is_mouse_pressed
cur_x = 0
cur_y = 0
is_mouse_pressed = False

reset_rotate = False

def mouse_down_callback(window, btn, action, mods):
    global init_x, init_y, cur_x, cur_y, is_mouse_pressed
    if (action == glfw.PRESS):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            # print("Pressed the left key!")
            init_x = cur_x
            init_y = cur_y
            is_mouse_pressed = True
    else:
        if btn == glfw.MOUSE_BUTTON_LEFT:
            # print("Released the left key!")
            is_mouse_pressed = False

def mouse_move_callback(window, x, y):
    global cur_x, cur_y
    cur_x = x
    cur_y = y

class mVisualBox():
    # basedir is the root dir of shader and model dirs
    # model_size: 0.04, 30.0
    def __init__(self, wnd_width, wnd_height, title="test", btn_callback=None, proj_mat=[], view_mat=[], limbs_n_root=[], model_size=0.04):
        self.wnd_width = wnd_width
        self.wnd_height = wnd_height
        self.app = MyApplication(title = "Visual Box", wndWidth=wnd_width, wndHeight=wnd_height)
        self.model_size = model_size

        assert(self.model_size in [0.04, 30.0])

        def button_callback(window, key, scancode, action, mods):
            global reset_rotate
            if (action != glfw.PRESS):
                return
            if key == glfw.KEY_R:
                reset_rotate = True
                return
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, GL_TRUE)
                return

            if btn_callback is not None:
                btn_callback.call(key)


        self.app.glfwInit(mouse_down_callback = mouse_down_callback, mouse_move_callback=mouse_move_callback, button_callback = button_callback)
        cam_shader = ShaderReader(os.path.join(SHADER_DIR, "v.shader"), os.path.join(SHADER_DIR, "f.shader"))
        model_shader = ShaderReader(os.path.join(SHADER_DIR, "v2.shader"), os.path.join(SHADER_DIR, "f2.shader"))

        if len(proj_mat) == 0:
            in_cammat_display = OpenGLUtils.perspective(np.radians(60), float(wnd_width) / wnd_height, 0.1, 10000.0)
        else:
            in_cammat_display = proj_mat

        if len(view_mat) == 0:
            ex_cammat_display = OpenGLUtils.lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0])
        else:
            ex_cammat_display = view_mat

        self.cam_scene = CamScene(cam_shader, wnd_width, wnd_height)

        mesh_render = MeshRenderer()
        mesh_render.addMesh(os.path.join(PLY_DIR, "sphere-{}.ply".format(self.model_size)))
        self.mpose_model = PoseModel(mesh_render, model_shader, (wnd_width, wnd_height), in_cammat_display, ex_cammat_display, limbs_n_root)

    def checkStop(self):
        return self.app.windowShouldClose()

    def setProjMat(self, proj_mat):
        self.mpose_model.setProjMat(proj_mat)
    def setViewMat(self, view_mat):
        self.mpose_model.setViewMat(view_mat)

    def clearStop(self):
        self.app.setWindowShouldClose(False)

    def draw(self, cam_bg, pose_list, color_list):
        for pose_i in range(len(pose_list)):
            self.drawPose(pose_list[pose_i], color_list[pose_i])

        if cam_bg is None:
            cam_bg = 51 * np.ones([self.wnd_width, self.wnd_height, 3], dtype=np.uint8)
        self.drawCam(cam_bg)

    def drawPose(self, joints_3d, color):
        self.mpose_model.draw(joints_3d, color, sphere_ratio=self.model_size)

    def drawCam(self, cam_bg):
        self.cam_scene.drawFrame(cam_bg)

    def begin(self):
        self.app.renderStart()
        global init_x, init_y, cur_x, cur_y, is_mouse_pressed
        ###################### handle the rotate event##################
        if is_mouse_pressed:
            div_x = cur_x - init_x
            div_y = cur_y - init_y

            if div_x != 0 or div_y != 0:
                # point_from = np.array([self.wnd_width - init_x, self.wnd_height - init_y, 0])
                # point_to = np.array([self.wnd_width - cur_x, self.wnd_height - cur_y, 0])

                point_from = np.array([self.wnd_width - init_x, 0, 0])
                point_to = np.array([self.wnd_width - cur_x, 0, 0])

                origin_p = (point_to + point_from) / 2
                origin_p[2] = - self.wnd_width / 10.0
                vec_from = OpenGLUtils.normalize(point_from - origin_p)
                vec_to = OpenGLUtils.normalize(point_to - origin_p)

                axis = OpenGLUtils.normalize(np.cross(vec_from, vec_to))
                theta = np.arccos(np.dot(vec_from, vec_to))

                rotateMat = OpenGLUtils.rotate(axis, theta, True)
                self.mpose_model.rotate(rotateMat)
        else:
            self.mpose_model.rotate(set_prev=True)
        #################################################################

    def end(self):
        global reset_rotate
        self.app.renderEnd()

        if reset_rotate:
            reset_rotate = False
            self.mpose_model.rotate(reset=True)

    def terminate(self):
        self.app.terminate()
