from Junction1Phase import Junction1Phase, Junction1PhaseParameters
from enums import ModelType
from error_utilities import error
from thermodynamic_functions import computeVolumeFraction, computeDensity, \
  computeSpecificVolume, computeVelocity, computeSpecificTotalEnergy, \
  computeSpecificInternalEnergy

class CompressibleJunctionParameters(Junction1PhaseParameters):
  def __init__(self):
    Junction1PhaseParameters.__init__(self)

## Junction that uses compressible flow assumption
class CompressibleJunction(Junction1Phase):
  def __init__(self, params):
    Junction1Phase.__init__(self, params)
    if self.n_meshes != 2:
      error("CompressibleJunction is only implemented for connecting 2 meshes.")

    # get node indices
    self.k_left = self.node_indices[0]
    self.k_right = self.node_indices[1]
    self.k_left_adjacent = self.adjacent_node_indices[0]
    self.k_right_adjacent = self.adjacent_node_indices[1]

    # get DoF indices
    if (self.model_type == ModelType.TwoPhase):
      self.i_vf1L = self.dof_handler.i(self.k_left, self.vf1_index)
      self.i_vf1R = self.dof_handler.i(self.k_right, self.vf1_index)
      self.i_vf1LL = self.dof_handler.i(self.k_left_adjacent, self.vf1_index)
      self.i_vf1RR = self.dof_handler.i(self.k_right_adjacent, self.vf1_index)

    self.i_arhoL = self.dof_handler.i(self.k_left, self.arho_index)
    self.i_arhouL = self.dof_handler.i(self.k_left, self.arhou_index)
    self.i_arhoEL = self.dof_handler.i(self.k_left, self.arhoE_index)

    self.i_arhoR = self.dof_handler.i(self.k_right, self.arho_index)
    self.i_arhouR = self.dof_handler.i(self.k_right, self.arhou_index)
    self.i_arhoER = self.dof_handler.i(self.k_right, self.arhoE_index)

    self.i_arhoLL = self.dof_handler.i(self.k_left_adjacent, self.arho_index)
    self.i_arhouLL = self.dof_handler.i(self.k_left_adjacent, self.arhou_index)
    self.i_arhoELL = self.dof_handler.i(self.k_left_adjacent, self.arhoE_index)

    self.i_arhoRR = self.dof_handler.i(self.k_right_adjacent, self.arho_index)
    self.i_arhouRR = self.dof_handler.i(self.k_right_adjacent, self.arhou_index)
    self.i_arhoERR = self.dof_handler.i(self.k_right_adjacent, self.arhoE_index)

  def applyWeaklyToNonlinearSystem(self, U, U_old, r, J):
    pass

  def applyStronglyToNonlinearSystem(self, U_new, U_old, r, J):
    n_values = 6
    L = 0
    R = 1
    L_old = 2
    R_old = 3
    LL_old = 4
    RR_old = 5

    k = [0] * n_values
    k[L] = self.k_left
    k[R] = self.k_right
    k[L_old] = self.k_left
    k[R_old] = self.k_right
    k[LL_old] = self.k_left_adjacent
    k[RR_old] = self.k_right_adjacent

    i_arho = [0] * n_values
    i_arho[L] = self.i_arhoL
    i_arho[R] = self.i_arhoR
    i_arho[L_old] = self.i_arhoL
    i_arho[R_old] = self.i_arhoR
    i_arho[LL_old] = self.i_arhoLL
    i_arho[RR_old] = self.i_arhoRR

    i_arhou = [0] * n_values
    i_arhou[L] = self.i_arhouL
    i_arhou[R] = self.i_arhouR
    i_arhou[L_old] = self.i_arhouL
    i_arhou[R_old] = self.i_arhouR
    i_arhou[LL_old] = self.i_arhouLL
    i_arhou[RR_old] = self.i_arhouRR

    i_arhoE = [0] * n_values
    i_arhoE[L] = self.i_arhoEL
    i_arhoE[R] = self.i_arhoER
    i_arhoE[L_old] = self.i_arhoEL
    i_arhoE[R_old] = self.i_arhoER
    i_arhoE[LL_old] = self.i_arhoELL
    i_arhoE[RR_old] = self.i_arhoERR

    U = [0] * n_values
    U[L] = U_new
    U[R] = U_new
    U[L_old] = U_old
    U[R_old] = U_old
    U[LL_old] = U_old
    U[RR_old] = U_old

    # initialize lists
    rho = [0] * n_values
    drho_dvf1 = [0] * n_values
    drho_darho = [0] * n_values

    u = [0] * n_values
    du_darho = [0] * n_values
    du_darhou = [0] * n_values

    e = [0] * n_values
    de_darho = [0] * n_values
    de_darhou = [0] * n_values
    de_darhoE = [0] * n_values

    p = [0] * n_values
    dp_dvf1 = [0] * n_values
    dp_darho = [0] * n_values
    dp_darhou = [0] * n_values
    dp_darhoE = [0] * n_values

    c = [0] * n_values
    dc_dvf1 = [0] * n_values
    dc_darho = [0] * n_values
    dc_darhou = [0] * n_values
    dc_darhoE = [0] * n_values

    # loop over subscript/superscript combinations
    for i in xrange(n_values):
      vf1 = self.dof_handler.getVolumeFraction(U[i], k[i])
      vf, dvf_dvf1 = computeVolumeFraction(vf1, self.phase, self.model_type)
      arho  = U[i][i_arho[i]]
      arhou = U[i][i_arhou[i]]
      arhoE = U[i][i_arhoE[i]]

      rho[i], drho_dvf, drho_darho[i] = computeDensity(vf, arho)
      drho_dvf1[i] = drho_dvf * dvf_dvf1

      u[i], du_darho[i], du_darhou[i] = computeVelocity(arho, arhou)

      v, dv_drho = computeSpecificVolume(rho[i])
      dv_dvf1 = dv_drho * drho_dvf1[i]
      dv_darho = dv_drho * drho_darho[i]

      E, dE_darho, dE_darhoE = computeSpecificTotalEnergy(arho, arhoE)

      e[i], de_du, de_dE = computeSpecificInternalEnergy(u[i], E)
      de_darho[i] = de_du * du_darho[i] + de_dE * dE_darho
      de_darhou[i] = de_du * du_darhou[i]
      de_darhoE[i] = de_dE * dE_darhoE

      p[i], dp_dv, dp_de = self.eos.p(v, e[i])
      dp_dvf1[i] = dp_dv * dv_dvf1
      dp_darho[i] = dp_dv * dv_darho + dp_de * de_darho[i]
      dp_darhou[i] = dp_de * de_darhou[i]
      dp_darhoE[i] = dp_de * de_darhoE[i]

      c[i], dc_dv, dc_de = self.eos.c(v, e[i])
      dc_dvf1[i] = dc_dv * dv_dvf1
      dc_darho[i] = dc_dv * dv_darho + dc_de * de_darho[i]
      dc_darhou[i] = dc_de * de_darhou[i]
      dc_darhoE[i] = dc_de * de_darhoE[i]

    # compute old average quantities
    rhoL = 0.5 * (rho[L_old] + rho[LL_old])
    rhoR = 0.5 * (rho[R_old] + rho[RR_old])
    uL = 0.5 * (u[L_old] + u[LL_old])
    uR = 0.5 * (u[R_old] + u[RR_old])
    pL = 0.5 * (p[L_old] + p[LL_old])
    pR = 0.5 * (p[R_old] + p[RR_old])
    cL = 0.5 * (c[L_old] + c[LL_old])
    cR = 0.5 * (c[R_old] + c[RR_old])

    # residual indices
    i1 = self.i_arhoL
    i2 = self.i_arhouL
    i3 = self.i_arhoEL
    i4 = self.i_arhoR
    i5 = self.i_arhouR
    i6 = self.i_arhoER

    # reset Jacobian rows
    J[i1,:] = 0
    J[i2,:] = 0
    J[i3,:] = 0
    J[i4,:] = 0
    J[i5,:] = 0
    J[i6,:] = 0

    # compute residuals and Jacobians
    r[i1] = p[L] - pL + rhoL * (cL - uL) * (u[L] - uL)
    J_i1_vf1L = dp_dvf1[L]
    J[i1,self.i_arhoL] = dp_darho[L] + rhoL * (cL - uL) * du_darho[L]
    J[i1,self.i_arhouL] = dp_darhou[L] + rhoL * (cL - uL) * du_darhou[L]
    J[i1,self.i_arhoEL] = dp_darhoE[L]

    r[i2] = p[R] - pR - rhoR * (cR - uR) * (u[R] - uR)
    J_i2_vf1R = dp_dvf1[R]
    J[i2,self.i_arhoR] = dp_darho[R] - rhoR * (cR - uR) * du_darho[R]
    J[i2,self.i_arhouR] = dp_darhou[R] - rhoR * (cR - uR) * du_darhou[R]
    J[i2,self.i_arhoER] = dp_darhoE[R]

    if u[L] >= 0:
      if u[R] < 0:
        error("Assumption violated: Both velocity conditions were true.")
      r[i3] = rho[L] - rhoL - (p[L] - pL) / cL**2
      J_i3_vf1L = drho_dvf1[L] - dp_dvf1[L] / cL**2
      J_i3_vf1R = 0
      J[i3,self.i_arhoL] = drho_darho[L] - dp_darho[L] / cL**2
      J[i3,self.i_arhouL] = - dp_darhou[L] / cL**2
      J[i3,self.i_arhoEL] = - dp_darhoE[L] / cL**2
    elif u[R] < 0:
      r[i3] = rho[R] - rhoR - (p[R] - pR) / cR**2
      J_i3_vf1R = drho_dvf1[R] - dp_dvf1[R] / cR**2
      J_i3_vf1L = 0
      J[i3,self.i_arhoR] = drho_darho[R] - dp_darho[R] / cR**2
      J[i3,self.i_arhouR] = - dp_darhou[R] / cR**2
      J[i3,self.i_arhoER] = - dp_darhoE[R] / cR**2
    else:
      error("Assumption violated: Neither velocity condition was true.")

    r[i4] = rho[L] * u[L] - rho[R] * u[R]
    J_i4_vf1L = drho_dvf1[L] * u[L]
    J[i4,self.i_arhoL] = drho_darho[L] * u[L] + rho[L] * du_darho[L]
    J[i4,self.i_arhouL] = rho[L] * du_darhou[L]
    J_i4_vf1R = -drho_dvf1[R] * u[R]
    J[i4,self.i_arhoR] = -(drho_darho[R] * u[R] + rho[R] * du_darho[R])
    J[i4,self.i_arhouR] = -rho[R] * du_darhou[R]

    r[i5] = e[L] + p[L] / rho[L] + 0.5 * u[L]**2 - (e[R] + p[R] / rho[R] + 0.5 * u[R]**2)
    J_i5_vf1L = dp_dvf1[L] / rho[L] - p[L] / rho[L]**2 * drho_dvf1[L]
    J[i5,self.i_arhoL] = de_darho[L] + dp_darho[L] / rho[L] - p[L] / rho[L]**2 * drho_darho[L] + u[L] * du_darho[L]
    J[i5,self.i_arhouL] = de_darhou[L] + dp_darhou[L] / rho[L] + u[L] * du_darhou[L]
    J[i5,self.i_arhoEL] = de_darhoE[L] + dp_darhoE[L] / rho[L]
    J_i5_vf1R = -(dp_dvf1[R] / rho[R] - p[R] / rho[R]**2 * drho_dvf1[R])
    J[i5,self.i_arhoR] = -(de_darho[R] + dp_darho[R] / rho[R] - p[R] / rho[R]**2 * drho_darho[R] + u[R] * du_darho[R])
    J[i5,self.i_arhouR] = -(de_darhou[R] + dp_darhou[R] / rho[R] + u[R] * du_darhou[R])
    J[i5,self.i_arhoER] = -(de_darhoE[R] + dp_darhoE[R] / rho[R])

    r[i6] = rho[L] * u[L]**2 + p[L] - (rho[R] * u[R]**2 + p[R])
    J_i6_vf1L = drho_dvf1[L] * u[L]**2 + dp_dvf1[L]
    J[i6,self.i_arhoL] = drho_darho[L] * u[L]**2 + rho[L] * 2.0 * u[L] * du_darho[L] + dp_darho[L]
    J[i6,self.i_arhouL] = rho[L] * 2.0 * u[L] * du_darhou[L] + dp_darhou[L]
    J[i6,self.i_arhoEL] = dp_darhoE[L]
    J_i6_vf1R = -(drho_dvf1[R] * u[R]**2 + dp_dvf1[R])
    J[i6,self.i_arhoR] = -(drho_darho[R] * u[R]**2 + rho[R] * 2.0 * u[R] * du_darho[R] + dp_darho[R])
    J[i6,self.i_arhouR] = -(rho[R] * 2.0 * u[R] * du_darhou[R] + dp_darhou[R])
    J[i6,self.i_arhoER] = -dp_darhoE[R]

    if self.model_type == ModelType.TwoPhase:
      J[i1,self.i_vf1L] = J_i1_vf1L
      J[i2,self.i_vf1R] = J_i2_vf1R
      J[i3,self.i_vf1L] = J_i3_vf1L
      J[i3,self.i_vf1R] = J_i3_vf1R
      J[i4,self.i_vf1L] = J_i4_vf1L
      J[i4,self.i_vf1R] = J_i4_vf1R
      J[i5,self.i_vf1L] = J_i5_vf1L
      J[i5,self.i_vf1R] = J_i5_vf1R
      J[i6,self.i_vf1L] = J_i6_vf1L
      J[i6,self.i_vf1R] = J_i6_vf1R

  def applyStronglyToLinearSystemMatrix(self, A):
    pass

  def applyStronglyToLinearSystemRHSVector(self, U_old, b):
    pass
