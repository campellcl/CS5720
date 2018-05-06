"""
open_gl_framework.py
"""
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import math
import numpy as np


class OpenGLFramework:
    button_down = None
    # image_width = None
    # image_height = None
    # image_depth = None
    # values = None

    def display(self):
        global eye, target, up, fov_y, aspect, near, far, vertices, points, normals

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov_y, aspect, near, far)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2],
                  target[0], target[1], target[2],
                  up[0], up[1], up[2])

        glLightfv(GL_LIGHT0, GL_POSITION, eye)

        color = [1.0, 1.0, 0.0, 1.]
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0)

        vertices = []
        points = []
        for i in range(image_width):
            for j in range(image_height):
                for k in range(image_depth):
                    # if self.values[i, j, k]:
                    if self.values[i]:
                        print(i, j, k)
                        points.append(([i, j, k], [0, 1, 0]))
                    else:
                        points.append(([i, j, k], [1, 0, 0]))
        points = np.array(points)

        glDisable(GL_LIGHTING)
        glBegin(GL_TRIANGLES)
        for i in range(len(vertices)):
            glColor3fv([1, 1, 1])
            glNormal3fv(normals[i, :])
            glVertex3fv(vertices[i, :])

        glEnd()

        glPointSize(10)
        glBegin(GL_POINTS)
        for point, c in points:
            glColor3fv(c)
            glVertex3fv(point)
        glEnd()
        glEnable(GL_LIGHTING)

        glutSwapBuffers()

    def motion_func(self, x, y):
        global win_id
        # this function modeled after modeler.PerspectiveCamera.orbit() function written by 'ags' here:
        # http://www.cs.cornell.edu/courses/cs4620/2008fa/asgn/model/model-fmwk.zip
        global previous_point, eye
        x *= 2 / window[0]
        y *= -2 / window[1]
        if self.button_down == 'left':
            mouse_delta = [x - previous_point[0], y - previous_point[1]]
            neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
            dist = sum([a**2 for a in neg_gaze]) ** (1/2)
            neg_gaze = [a / dist for a in neg_gaze]
            azimuth = math.atan2(neg_gaze[0], neg_gaze[2])
            elevation = math.atan2(neg_gaze[1], (neg_gaze[0]**2 + neg_gaze[2]**2)**(1/2))
            azimuth = (azimuth - mouse_delta[0]) % (2 * math.pi)
            elevation = max(-math.pi * .495, min(math.pi * .495, elevation - mouse_delta[1]))
            neg_gaze[0] = math.sin(azimuth) * math.cos(elevation)
            neg_gaze[1] = math.sin(elevation)
            neg_gaze[2] = math.cos(azimuth) * math.cos(elevation)
            mag = sum([a**2 for a in neg_gaze]) ** (1/2)
            neg_gaze = [a / mag * dist for a in neg_gaze]
            new_eye = [a + b for a, b in zip(target, neg_gaze)]
            eye = new_eye
            glutPostRedisplay()
        elif self.button_down == 'right':
            mouse_delta_y = y - previous_point[1]
            neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
            dist = sum([a**2 for a in neg_gaze]) ** (1/2)
            new_dist = dist * 2 ** (mouse_delta_y)
            new_neg_gaze = [a / dist * new_dist for a in neg_gaze]
            new_eye = [a + b for a, b in zip(target, new_neg_gaze)]
            eye = new_eye
            glutPostRedisplay()

        # print(eye)
        previous_point = (x, y)

    def mouse_func(self, button, state, x, y):
        global previous_point, button_down
        # print(button_down, state, x, y)
        previous_point = (x * 2 / window[0], -y * 2 / window[1])
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                button_down = 'left'
            elif state == GLUT_UP:
                button_down = None
        elif button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                button_down = 'right'
            elif state == GLUT_UP:
                button_down = None

    def __init__(self, x, y, z, values):
        global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id
        image_width = len(x)
        image_height = len(y)
        image_depth = len(z)
        self.values = values
        eye = [(image_width-1)/2, (image_height-1)/2, 2*image_depth]
        target = [(image_width-1)/2, (image_height-1)/2, (image_depth-1)/2]
        up = [0, 1, 0]

        window = (800, 800)
        fov_y = 40
        near = .1
        far = 1000

        aspect = window[0] / window[1]
        light_position = eye
        light_color = [100.0, 100.0, 100.0, 1.0]

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(window[0], window[1])
        win_id = glutCreateWindow(b'3D Points')

        glClearColor(0., 0., 0., 1.)
        glShadeModel(GL_SMOOTH)
        # glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)

        glEnable(GL_PROGRAM_POINT_SIZE)

        glEnable(GL_LIGHT0)

        # callbacks
        glutDisplayFunc(self.display)
        glutMouseFunc(self.mouse_func)
        glutMotionFunc(self.motion_func)

        glutMainLoop()
