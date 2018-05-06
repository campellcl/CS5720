from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys

name = 'ball_glut'

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(400,400)
    glutCreateWindow(name)

    glClearColor(0.,0.,0.,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    lightZeroPosition = [10.,4.,10.,1.]
    lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    glutDisplayFunc(display)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(40.,1.,1.,40.)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(5,5,10,
              0,0,0,
              0,1,0)
    glPushMatrix()
    glutMainLoop()
    return

def display():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    color = [1.0,0.,0.,1.]
    glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
    # glutSolidSphere(2,20,20)
    # To draw a cube we must specify where the vertices are. We will label a,b,c,d for the bottom square going counter
    # clockwise.
    # we need to know which direction our axis are. We will use to the right is x, up is y, and z is 3d.
    a = (1, -1, -1)
    b = (-1, -1, -1)
    c = (-1, -1, 1)
    d = (1, -1, 1)
    e = (1, 1, -1)
    f = (-1, 1, -1)
    g = (-1, 1, 1)
    h = (1, 1, 1)
    # We also need to keep track of the normals (a vector pointing perpendicular to the face of the cube)
    # To do this we must name the faces on the cube:
    right_face = (1, 0, 0)
    left_face = (-1, 0, 0)
    top_face = (0, 1, 0)
    bottom_face = (0, -1, 0)
    front_face = (0, 0, 1)
    back_face = (0, 0, -1)
    # Now we need to use the above vars to create triangles:
    # When we specify the vertices for the traingle the order matters. We use the righthand rule. for example: tri = (d,h,g)
    triangles = [(d,h,g),(c,d,g)]
    # To render:
    glBegin(GL_TRIANGLES)
    glNormal3fv(front_face)
    glVertex3fv(d)
    glVertex3fv(h)
    glVertex3fv(g)
    color = [0.0,1.0,0.0,1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
    glVertex3fv(c)
    glVertex3fv(d)
    glVertex3fv(g)
    color = [0.0,0.0,1.0,1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
    glNormal3fv(right_face)
    glVertex3fv(d)
    glVertex3fv(a)
    glVertex3fv(h)
    color = [.5,.5,0.0,1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
    glVertex3fv(a)
    glVertex3fv(e)
    glVertex3fv(h)
    glEnd()
    glPopMatrix()
    glutSwapBuffers()
    return

if __name__ == '__main__': main()