import numpy as np
from utils import get_num_cols_rows
import sympy
import math


def quaternion_matrix(quaternion):
  """Return homogeneous rotation matrix from quaternion.

  >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
  >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
  True
  >>> M = quaternion_matrix([1, 0, 0, 0])
  >>> np.allclose(M, np.identity(4))
  True
  >>> M = quaternion_matrix([0, 1, 0, 0])
  >>> np.allclose(M, np.diag([1, -1, -1, 1]))
  True

  """
  _EPS = np.finfo(float).eps * 4.0
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)
  if n < _EPS:
    return np.identity(4)
  q *= math.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
    [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
    [                0.0,                 0.0,                 0.0, 1.0]])


def get_transform_matrix(pose):
  translation = pose[1:4]
  transform_matrix = quaternion_matrix(pose[4:])
  transform_matrix[:-1, -1] = np.asarray(translation).T
  return transform_matrix


# def create_equation(line_equation, transform_matrix, cx, cy, fx, fy):
#   a, b, c, d = line_equation
#   Z, u, v = sympy.symbols('Z u v')
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#   vect = sympy.Matrix([X, Y, Z, 1])
#   coord = transform_matrix * vect
#   equation = a * coord[0] + b * coord[1] + c * coord[2] - d
#   return X, Y, Z, equation

def convert_pix_to_3d_point(u, v, plane_equation, transform_matrix, cx, cy, fx, fy):
  a, b, c, d = plane_equation
  Z = sympy.symbols('z')
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  vect = sympy.Matrix([X, Y, Z, 1])
  coord = transform_matrix * vect
  equation = a * coord[0] + b * coord[1] + c * coord[2] - d
  Z_val = sympy.solvers.solve(equation, Z)[0]
  X_val = X.evalf(subs={Z: Z_val})
  Y_val = Y.evalf(subs={Z: Z_val})
  return X_val, Y_val, Z_val


def convert_3d_point_to_pix(X, Y, Z, transform_matrix, cx, cy, fx, fy):
  u = fx * X / Z + cx
  Y = (v - cy) * Z / fy
  y =
