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


def quaternion_inverse(quaternion):
  """Return inverse of quaternion.

  >>> q0 = random_quaternion()
  >>> q1 = quaternion_inverse(q0)
  >>> numpy.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
  True

  """
  q = np.array(quaternion, dtype=np.float64, copy=True)
  np.negative(q[1:], q[1:])
  return q / np.dot(q, q)

def get_camera_to_world(translation, rotation):
  '''
  Camera to World transform matrix
  :param pose: camera pose
  :return:
  '''
  # transform_matrix = get_world_to_camera(translation, rotation)
  # transform_matrix[0:3, 0:3] = transform_matrix[0:3, 0:3].T
  # transform_matrix[:-1, -1] = -transform_matrix[:-1, -1]
  # return transform_matrix
  return np.linalg.inv(get_world_to_camera(translation, rotation))

def get_world_to_camera(translation, rotation):
  '''
  World to Camera transform matrix
  :param pose: camera pose
  :return:
  '''
  transform_matrix = np.asarray(
    [[rotation[0], rotation[1], rotation[2], translation[0]],
     [rotation[3], rotation[4], rotation[5], translation[1]],
     [rotation[6], rotation[7], rotation[8], translation[2]],
     [0, 0, 0, 1]]
  )
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


# def convert_pix_to_3d_point(u, v, plane_equation, camera_to_world, fx, fy, cx, cy):
#   a, b, c, d = plane_equation
#   # X, Y, Z in camera coordinate
#   Z = sympy.symbols('z')
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#   vect = sympy.Matrix([X, Y, Z, 1])
#   # convert X, Y, Z to world coordinate by using Camera-To-World matrix
#   coord = camera_to_world * vect
#   # finally substitue to plane equation
#   equation = a * coord[0] + b * coord[1] + c * coord[2] - d
#   Z_val = sympy.solvers.solve(equation, Z)[0]
#   X_val = X.evalf(subs={Z: Z_val})
#   Y_val = Y.evalf(subs={Z: Z_val})
#   # return 3d coordinate in world
#   return X_val, Y_val, Z_val

def convert_pix_to_3d_point(u, v, plane_equation, camera_to_world, fx, fy, cx, cy, z_max=float('inf')):
  a, b, c, d = plane_equation
  # X, Y, Z in camera coordinate
  z = sympy.symbols('z')
  x = (u - cx) * z / fx
  y = (v - cy) * z / fy
  vect = sympy.Matrix([x, y, z, 1])
  # convert X, Y, Z to world coordinate by using Camera-To-World matrix
  X, Y, Z, _ = camera_to_world * vect
  # finally substitue to plane equation
  equation = a * X + b * Y + c * Z + d
  z_val = sympy.solvers.solve(equation, z)[0]
  X_val = X.evalf(subs={z: z_val})
  Y_val = Y.evalf(subs={z: z_val})
  Z_val = Z.evalf(subs={z: z_val})
  # return 3d coordinate in world
  if z_val > 0 and z_val < z_max:
    return X_val, Y_val, Z_val
  else:
    return None


# def distort(x, y, fx, fy, cx, cy, k1, k2, p1, p2, k3):
#   r = ((x - cx) / fx) ** 2 + ((y - cy) / fy) ** 2
#   radial_distortion = (1 + k1 * r + k2 * (r ** 2) + k3 * (r ** 3))
#   x = x * radial_distortion
#   y = y * radial_distortion
#   x = x + 2 * p1 * x * y + p2 * (r ** 2 + 2 * (x ** 2))
#   y = y + p1 * (r ** 2 + 2 * (y ** 2)) + 2 * p2 * x * y
#   return x, y

def convert_3d_point_to_pix(X, Y, Z, world_to_camera, fx, fy, cx, cy):
  # X, Y, Z in world coordinate
  # convert them to camera coordinate first
  vect = np.asarray([X, Y, Z, 1])
  # X, Y, Z = coordinate in camera
  X, Y, Z, _ = world_to_camera.dot(vect)
  # finally compute u, v
  # u = (fx * X + world_to_camera[0, 3]) / Z + cx
  # v = (fy * Y + world_to_camera[1, 3]) / Z + cy
  u = fx * X / Z + cx
  v = fy * Y / Z + cy
  # return pixel coordinate
  return int(u), int(v)

# pose = [0.1, 0.2, 0.25, 1.3, 0.7, 0.2, 0.15]
# y = get_inverse_transform_matrix(pose)