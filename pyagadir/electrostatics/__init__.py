"""
Electrostatics of the helix-coil model.

  point.py              point charges at tabulated distances, screened Coulomb in water: the side chain-side chain and
                        terminal-group pairs and the helix macrodipole (a +1/2 and a -1/2 charge at the helix ends).
  common.py             the self-consistent ionisation solver and the electrostatic energy terms.
  side_chain_states.py  the acid-base side chains as states (contacts at i+-3/i+-4 or a capping motif), summed by a
                        transfer matrix.

Electrostatics (common.py) is a mixin of energies.PrecomputeParams.
"""
from pyagadir.electrostatics.common import Electrostatics
from pyagadir.electrostatics.point import PointChargeElectrostatics

__all__ = ["Electrostatics", "PointChargeElectrostatics"]
