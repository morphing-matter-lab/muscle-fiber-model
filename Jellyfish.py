import potpourri3d as pp3d
import igl
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation as R


def dot_vec(a, b):   # vector-vector dot product
  return np.sum(a * b, axis=1)[:, None]

def squared_norm(v):
  return np.dot(v, v)

def qmultiply(q1, q2):
  q1_s = q1[3]
  q2_s = q2[3]
  q1_v = q1[:3]
  q2_v = q2[:3]
  
  real = q1_s * q2_s - np.dot(q1_v, q2_v)
  imag = q1_s * q2_v + q2_s * q1_v + np.cross(q1_v, q2_v)
  return np.array([imag[0], imag[1], imag[2], real])

def qbar(q):
  return np.array([-q[0], -q[1], -q[2], q[3]])
  
def qinv(q):
  norm = squared_norm(q)
  if norm == 0:
      print("Error: division by zero")
  return qbar(q) / norm

class Jellyfish:
  curr_verts = None
  curr_normals = None
  curr_masses = None

  next_verts = None
  next_normals = None
  next_masses = None
  
  epsilon = 0.1
  alpha = epsilon
  beta = 1 - epsilon

  # Following the paper's notation, g is an element of SE(3) represented as a unit quaternion for rotation and 3d vector for translation
  curr_g = np.concatenate((np.array([0, 0, 0, 1]), np.array([0, 0, 0])))

  def __init__(self, verts, faces):
    self.curr_verts = verts
    self.curr_normals = igl.per_vertex_normals(verts, faces)
    self.curr_masses = pp3d.vertex_areas(verts, faces).reshape(-1, 1)

  def force(self, next_verts, next_normals, next_masses):
    # compute forces
    ΔP = next_verts - self.curr_verts
    F1 = self.alpha * ΔP + self.beta * dot_vec(ΔP, self.curr_normals) * self.curr_normals
    F2 = self.alpha * ΔP + self.beta * dot_vec(ΔP, next_normals) * next_normals

    # masses
    F1 = self.curr_masses * F1
    F2 = next_masses * F2

    return -sum(F1 + F2)

  def torque(self, next_verts, next_normals, next_masses):
    # compute forces
    ΔP = next_verts - self.curr_verts
    F1 = self.alpha * ΔP + self.beta * dot_vec(ΔP, self.curr_normals) * self.curr_normals
    F2 = self.alpha * ΔP + self.beta * dot_vec(ΔP, next_normals) * next_normals

    # compute torques
    T1 = np.cross(self.curr_verts, F2)
    T2 = np.cross(next_verts, F1)

    # masses
    T1 = self.curr_masses * T1
    T2 = next_masses * T2

    return -sum(T1 + T2)

  def fun(self, g):
    quat = g[:4]
    translation = g[4:]
    rot = R.from_quat(quat)

    next_verts = rot.apply(self.next_verts) + translation
    next_normals = rot.apply(self.next_normals)

    delta_force = self.force(next_verts, next_normals, self.next_masses)
    delta_torque = self.torque(next_verts, next_normals, self.next_masses)
    unit_length_constraint = np.array([squared_norm(quat) - 1])

    return np.concatenate((delta_force, delta_torque, unit_length_constraint))

  def next_frame(self, verts, faces):
    self.next_verts = verts
    self.next_normals = igl.per_vertex_normals(verts, faces)
    self.next_masses = pp3d.vertex_areas(verts, faces).reshape(-1, 1)

    # print(self.fun(self.curr_g))
    sol = optimize.root(self.fun, self.curr_g, method='krylov', options={'ftol': 1e-6} )
    self.curr_g = sol.x
    # print(sol.x)

    # normalize quaternion
    self.curr_g[:4] = self.curr_g[:4] / np.linalg.norm(self.curr_g[:4])

    # apply rigid motion 
    quat = self.curr_g[:4]
    translation = self.curr_g[4:]
    rot = R.from_quat(quat)

    self.curr_verts = self.next_verts
    self.curr_verts = rot.apply(self.curr_verts) + translation
    self.curr_normals = self.next_normals
    self.curr_normals = rot.apply(self.curr_normals)
    self.curr_masses = self.next_masses

