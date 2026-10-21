"""Throughput benchmark for the Stage 0 system (N149).

Usage: .venv/bin/python scripts/benchmark.py SEQ FF DEVICE DT_FS [PS]
DT_FS 4 uses hydrogen mass repartitioning (3 amu); 2 uses H-bond constraints only.
Prints one tab-separated line: seq ff device dt_fs atoms ns_per_day.
"""
import sys
import time

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
from openmm.app import PME, HBonds, Simulation

from build import build

seq, ff, device, dt_fs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ps = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0

top, pos, forcefield = build(seq, ff)
system = forcefield.createSystem(top, nonbondedMethod=PME, nonbondedCutoff=1.2 * unit.nanometer,
                                 switchDistance=1.0 * unit.nanometer, constraints=HBonds,
                                 hydrogenMass=(3 if dt_fs == 4 else 1) * unit.amu)
system.addForce(MonteCarloBarostat(1 * unit.bar, 278 * unit.kelvin))
integ = LangevinMiddleIntegrator(278 * unit.kelvin, 1 / unit.picosecond, dt_fs * unit.femtosecond)
plat = Platform.getPlatformByName("CUDA")
sim = Simulation(top, system, integ, plat, {"DeviceIndex": device, "Precision": "mixed"})
sim.context.setPositions(pos)
sim.minimizeEnergy(maxIterations=500)
sim.step(2000)  # warm-up
n = int(ps * 1000 / dt_fs)
t0 = time.time()
sim.step(n)
sim.context.getState(getEnergy=True)  # sync
dt = time.time() - t0
print(f"{seq}\t{ff}\tgpu{device}\t{dt_fs}\t{top.getNumAtoms()}\t{ps / 1000 / dt * 86400:.0f}", flush=True)
