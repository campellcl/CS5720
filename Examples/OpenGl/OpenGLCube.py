import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
)

triangles = (
    (0, 1, 2)
)


def Cube():
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.0)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def triangle():
    glBegin(GL_TRIANGLES)
    glNormal3f(0.0, 0.0, -1.0)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3fv(vertices[0])
    glColor3f(0.0, 1.0, 0.0)
    glVertex3fv(vertices[1])
    glColor3f(0.0, 0.0, 1.0)
    glVertex3fv(vertices[2])
    glEnd()


def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)
    # glEnable(GL_LIGHTING)
    # glLightModelfv(GL_LIGHT0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        #glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # Create light source:
        glLightfv(GL_LIGHT0, GL_POSITION, [5,5,5])
        Cube()
        triangle()
        pygame.display.flip()
        pygame.time.wait(10)


main()
