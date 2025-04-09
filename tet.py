import pyvista as pv
import tetgen
import numpy as np
import fabsim_py

# pv.set_plot_theme('document')

mesh = pv.read("tissue.obj")
tet = tetgen.TetGen(mesh)
V, F = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

fixed_dofs = []

def fix_dofs_in_circle(radius, centerX, centerY):
  center = np.array([centerX, centerY])
  for i in range(V.shape[0]):
    if np.linalg.norm(center - V[i, :2]) < radius + 1e-4:
      fixed_dofs.append(3 * i)
      fixed_dofs.append(3 * i + 1)
      fixed_dofs.append(3 * i + 2)

fix_dofs_in_circle(0.75, -4.55/2, 0)
fix_dofs_in_circle(0.75, 4.55/2, 0)

print(len(fixed_dofs))
fixed = np.zeros(V.shape[0])
for idx in fixed_dofs:
  if idx % 3 == 0:
    fixed[idx // 3] = 1

import polyscope as ps
ps.init()
ps_vol = ps.register_volume_mesh("test volume mesh", V, tets=F)
ps_vol.add_scalar_quantity("fixed", fixed, defined_on="vertices")
ps.show()

NV = V.copy()
fabsim_py.simulate3D(NV, V, F, fixed_dofs, 1.3, 0.3, 0)
fabsim_py.simulate3D(NV, V, F, fixed_dofs, 1.4, 0.3, 0)
fabsim_py.simulate3D(NV, V, F, fixed_dofs, 1.5, 0.3, 0)
fabsim_py.simulate3D(NV, V, F, fixed_dofs, 1.6, 0.3, 0)
ps_vol = ps.register_volume_mesh("sim volume mesh", NV, tets=F)
ps_vol.add_scalar_quantity("fixed", fixed, defined_on="vertices")
ps.show()
