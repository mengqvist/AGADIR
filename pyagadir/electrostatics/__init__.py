"""
Electrostatics of the helix-coil model. The law is chosen per prediction with dipole_law (AGADIR(dipole_law=...),
EnergyCalculator(dipole_law=...)):

  point.py   "munoz" (default): point charges at tabulated distances, screened Coulomb, and the side chain-macrodipole
             law of Muñoz & Serrano 1995. Also the terminal groups' terms for every law.
  gb.py      "gb", "gbx", "gbp" (opt-in, in development): Generalized Born, tabulated from built helices and coil
             chains, with the Table 1 values and parameters transferred to it (gbx, gbp) and the GB pair law (gbp).
  common.py  the switch between them, the self-consistent ionisation solver, and the energy terms both laws feed.

Each module holds a mixin of energies.PrecomputeParams, which inherits Electrostatics (common.py).
"""
from pyagadir.electrostatics.common import DIPOLE_LAWS, Electrostatics
from pyagadir.electrostatics.gb import GB_LAWS, GBX_LAWS, GBElectrostatics
from pyagadir.electrostatics.point import POINT_LAW, PointChargeElectrostatics

__all__ = ["DIPOLE_LAWS", "GB_LAWS", "GBX_LAWS", "POINT_LAW", "Electrostatics", "GBElectrostatics",
           "PointChargeElectrostatics"]
