import numpy as np


def PCA(data, correlation = False, sort = True):
  '''Applies Principal Component Analysis to the data

  Parameters
  ----------
  data: array
      The array containing the data. The array must have NxM dimensions, where each
      of the N rows represents a different individual record and each of the M columns
      represents a different variable recorded for that individual record.
          array([
          [V11, ... , V1m],
          ...,
          [Vn1, ... , Vnm]])

  correlation(Optional) : bool
          Set the type of matrix to be computed (see Notes):
              If True compute the correlation matrix.
              If False(Default) compute the covariance matrix.

  sort(Optional) : bool
          Set the order that the eigenvalues/vectors will have
              If True(Default) they will be sorted (from higher value to less).
              If False they won't.
  Returns
  -------
  eigenvalues: (1,M) array
      The eigenvalues of the corresponding matrix.

  eigenvector: (M,M) array
      The eigenvectors of the corresponding matrix.

  Notes
  -----
  The correlation matrix is a better choice when there are different magnitudes
  representing the M variables. Use covariance matrix in other cases.

  '''

  mean = np.mean(data, axis=0)

  data_adjust = data - mean

  #: the data is transposed due to np.cov/corrcoef syntax
  if correlation:

      matrix = np.corrcoef(data_adjust.T)

  else:
      matrix = np.cov(data_adjust.T)

  eigenvalues, eigenvectors = np.linalg.eig(matrix)

  if sort:
      #: sort eigenvalues and eigenvectors
      sort = eigenvalues.argsort()[::-1]
      eigenvalues = eigenvalues[sort]
      eigenvectors = eigenvectors[:,sort]

  return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=True):
  '''Computes the best fitting plane of the given points

  Parameters
  ----------
  points: array
      The x,y,z coordinates corresponding to the points from which we want
      to define the best fitting plane. Expected format:
          array([
          [x1,y1,z1],
          ...,
          [xn,yn,zn]])

  equation(Optional) : bool
          Set the oputput plane format:
              If True return the a,b,c,d coefficients of the plane.
              If False(Default) return 1 Point and 1 Normal vector.
  Returns
  -------
  a, b, c, d : float
      The coefficients solving the plane equation.

  or

  point, normal: array
      The plane defined by 1 Point and 1 Normal vector. With format:
      array([Px,Py,Pz]), array([Nx,Ny,Nz])

  '''

  w, v = PCA(points)

  #: the normal of the plane is the last eigenvector
  normal = v[:,2]

  #: get a point from the plane
  point = np.mean(points, axis=0)


  if equation:
      a, b, c = normal
      d = -(np.dot(normal, point))
      return a, b, c, d

  else:
      return point, normal


def compute_z(x, y, equation):
  a, b, c, d = equation
  if c == 0:
    return float('nan')
  else:
    return (d - a * x - b * y) / c


def compute_distance(point, equation):
  a, b, c, d = equation
  x, y, z = point
  return float(abs(a * x + b * y + c * z - d)) / (a * a + b * b + c * c)


def fit_plane_with_outlier_removed(points, distance_ratio=2):
  equation = best_fitting_plane(points, True)
  distances = []
  for point in points:
    distances.append(compute_distance(point, equation))
  distances = np.asarray(distances)
  std = np.std(distances)
  mean_distance = distances.mean()
  print('STD: ', std)
  print('Mean: ', mean_distance)
  accepted_points = []
  for j, point in enumerate(points):
    if abs(distances[j] - mean_distance) <= std * distance_ratio:
      accepted_points.append(point)
  accepted_points = np.asarray(accepted_points)
  print(len(accepted_points))
  if len(accepted_points) > 2:
    return best_fitting_plane(accepted_points)
  else:
    return equation
