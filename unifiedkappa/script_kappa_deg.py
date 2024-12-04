import sys
from crystal import class_poscar
from conductivity import class_kappa
import numpy as np


obj_poscar = class_poscar("./POSCAR-prim")
obj_kappa = class_kappa(obj_poscar)
obj_kappa.get_kappa_phonopy(
    mesh_in = [8,8,8],
    sc_mat = np.eye(3)*2,
    pm_mat = np.eye(3),
    list_temp = [300],
    name_pcell = "POSCAR-prim",
    name_ifc2nd = "FORCE_CONSTANTS_2ND",
    is_minikappa = False,
    is_planckian = False,
    is_sbtetau = True,
    path_sbtetau = "./",
    list_taufactor = [2.0]
)
