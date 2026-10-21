"""Stage 1b production run (N153): one job directory, on any machine with OpenMM >= 8.2.

Usage: python s1b_run.py JOBDIR [--device 0] [--platform CUDA] [--chunk-ns 2]
                                [--stop-at EPOCH_SECONDS | --max-hours H] [--until-ns NS] [--bench MINUTES]
JOBDIR is built by s1b_prepare.py (system.xml.gz, start.pdb, job.json). Needs only openmm: the cloud
notebooks install nothing else.

Protocol (N153 Design; Stage 0 settings): minimise, 1 ns NVT, then NPT at 278 K and 1 bar (Langevin middle,
1/ps; Monte Carlo barostat), 4 fs with HMR. Peptide-only XTC every 20 ps.

Portable restarts: a Stage 0 CUDA checkpoint cannot be loaded on a different GPU model, so this script saves
platform-independent XML States. Production runs in chunks (default 2 ns):
  traj/chunk_NNNN.xtc  frames of chunk N (N >= 1); time = (N - 1) * chunk_ns + frame * 20 ps
  state_NNNN.xml       the State at the end of chunk N (state_0000.xml: after NVT; kept). Written after the
                       chunk's XTC is complete, through a temporary file and a rename.
  progress.log         one line per chunk: UTC time, ns done, ns/day, device name; DONE at the end.
On start the script resumes from the highest state_NNNN.xml and deletes any XTC chunk beyond it (a chunk
interrupted mid-way is rerun from its start). A job directory can therefore move between machines: copy
the whole directory. Never run one job in two places at once.
Chunk size is fixed per job by its first production chunk (recorded in progress.log) so that chunk times
stay uniform; --chunk-ns only applies to a fresh job.

Stopping: --stop-at (local: the owner's 21:00 UK limit) or --max-hours (cloud session limits). Each chunk
starts with its first 20 ps frame as a timing probe; if the chunk would end after the deadline, that frame
is dropped, PAUSED is written and the script exits 0, so no chunk runs past the deadline. (Only a fresh
job's 1 ns NVT is not probed: s1b_queue.py does not start a job with less than 20 min left.) The job
resumes on the next call.
--until-ns NS stops once NS of production exist and writes HELD (not DONE), e.g. for a pilot that may later be
extended to the job's full length by a call without it.

--bench MINUTES: throughput only. Runs the job's system from start.pdb in memory (minimise, 20 ps NVT, then
NPT in 0.5 ns chunks) for MINUTES, prints ns/day per chunk and a summary line, and writes nothing to JOBDIR
except bench.log.
"""
import argparse
import glob
import gzip
import json
import os
import socket
import sys
import time

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
from openmm.app import PDBFile, Simulation, XTCReporter

ap = argparse.ArgumentParser()
ap.add_argument("jobdir")
ap.add_argument("--device", default="0")
ap.add_argument("--platform", default="CUDA")
ap.add_argument("--chunk-ns", type=float, default=2.0)
ap.add_argument("--stop-at", type=float, help="UTC epoch seconds; no chunk may end after it")
ap.add_argument("--max-hours", type=float, help="wall-clock budget for this call")
ap.add_argument("--bench", type=float, help="minutes of throughput measurement (writes only bench.log)")
ap.add_argument("--until-ns", type=float, help="stop (HELD, not DONE) once this much production exists")
args = ap.parse_args()

t_start = time.time()
deadline = args.stop_at or (t_start + 3600 * args.max_hours if args.max_hours else float("inf"))
J = json.load(open(os.path.join(args.jobdir, "job.json")))
T = J["temperature_K"] * unit.kelvin
DT = J["dt_fs"] * unit.femtosecond
FRAME_STEPS = int(round(J["frame_ps"] * 1000 / J["dt_fs"]))
LOG = os.path.join(args.jobdir, "bench.log" if args.bench else "progress.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC  {msg}"
    print(f"[{J['job']}] {line}", flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def make_simulation(barostat, seed):
    with gzip.open(os.path.join(args.jobdir, "system.xml.gz"), "rt") as f:
        system = XmlSerializer.deserialize(f.read())
    if barostat:
        b = MonteCarloBarostat(J["pressure_bar"] * unit.bar, T)
        b.setRandomNumberSeed(seed)
        system.addForce(b)
    integ = LangevinMiddleIntegrator(T, J["friction_per_ps"] / unit.picosecond, DT)
    integ.setRandomNumberSeed(seed)
    plat = Platform.getPlatformByName(args.platform)
    props = {"DeviceIndex": args.device, "Precision": "mixed"} if args.platform in ("CUDA", "OpenCL") else {}
    top = PDBFile(os.path.join(args.jobdir, "start.pdb")).topology
    sim = Simulation(top, system, integ, plat, props)
    try:
        dev = plat.getPropertyValue(sim.context, "DeviceName")
    except Exception:
        dev = args.platform
    return sim, top, dev


def peptide_atoms(top):
    return [a.index for a in top.atoms() if a.residue.chain.index == 0]


def save_state(sim, k):
    path = os.path.join(args.jobdir, f"state_{k:04d}.xml")
    tmp = path + ".tmp"
    st = sim.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                              enforcePeriodicBox=False)
    with open(tmp, "w") as f:
        f.write(XmlSerializer.serialize(st))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    for old in sorted(glob.glob(os.path.join(args.jobdir, "state_*.xml")))[:-2]:  # keep 0000 and the last two
        if not old.endswith("state_0000.xml"):
            os.remove(old)


def equilibrate(sim, top, seed):
    """Minimise, 1 ns NVT; returns the positions/velocities to continue from under NPT."""
    pos = PDBFile(os.path.join(args.jobdir, "start.pdb")).positions
    sim.context.setPositions(pos)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(T, seed)
    sim.step(int(round(J["nvt_ns"] * 1e6 / J["dt_fs"])))
    return sim.context.getState(getPositions=True, getVelocities=True)


def bench():
    sim, top, dev = make_simulation(False, J["run_seed"])
    sim.context.setPositions(PDBFile(os.path.join(args.jobdir, "start.pdb")).positions)
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(T, J["run_seed"])
    sim.step(5000)  # 20 ps NVT
    st = sim.context.getState(getPositions=True, getVelocities=True)
    sim, top, dev = make_simulation(True, J["run_seed"])
    sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    sim.context.setPositions(st.getPositions())
    sim.context.setVelocities(st.getVelocities())
    sim.reporters.append(_NullReporter())
    log(f"BENCH start {J['job']} atoms={top.getNumAtoms()} device={dev} host={socket.gethostname()}")
    rates, end = [], t_start + 60 * args.bench
    while time.time() < end:
        t0 = time.time()
        sim.step(int(0.5e6 / J["dt_fs"]))
        sim.context.getState(getEnergy=True)
        rates.append(0.5 / (time.time() - t0) * 86400)
        log(f"BENCH {0.5 * len(rates):g} ns  {rates[-1]:.0f} ns/day")
    later = rates[len(rates) // 2:] or rates
    log(f"BENCH SUMMARY {J['job']} atoms={top.getNumAtoms()} device={dev} host={socket.gethostname()} "
        f"chunks={len(rates)} first={rates[0]:.0f} last_half_mean={sum(later) / len(later):.0f} ns/day")


class _NullReporter:
    """Asks for positions every 20 ps, as production's XTC does, and discards them (bench parity)."""

    def describeNextReport(self, simulation):
        steps = FRAME_STEPS - simulation.currentStep % FRAME_STEPS
        return {"steps": steps, "periodic": True, "include": ["positions"]}

    def report(self, simulation, state):
        state.getPositions(asNumpy=True)


def run():
    traj = os.path.join(args.jobdir, "traj")
    os.makedirs(traj, exist_ok=True)
    for tmp in glob.glob(os.path.join(args.jobdir, "state_*.xml.tmp")):
        os.remove(tmp)
    states = sorted(glob.glob(os.path.join(args.jobdir, "state_*.xml")))
    k = int(os.path.basename(states[-1])[6:10]) if states else -1
    if os.path.exists(LOG) and open(LOG).read().rstrip().endswith("DONE"):
        print(f"[{J['job']}] already DONE", flush=True)
        return
    # the chunk length is fixed by the job's first production chunk
    chunk_ns = args.chunk_ns
    if os.path.exists(LOG):
        for line in open(LOG):
            if "chunk_ns=" in line:
                chunk_ns = float(line.split("chunk_ns=")[1].split()[0])
    n_chunks = int(round(J["ns"] / chunk_ns))
    for f in glob.glob(os.path.join(traj, "chunk_*.xtc")):
        if int(os.path.basename(f)[6:10]) > k:
            os.remove(f)  # written after the last saved state: rerun that chunk

    if k < 0:  # fresh job: minimise and NVT, then switch on the barostat
        sim, top, dev = make_simulation(False, J["run_seed"])
        log(f"START {J['job']} {J['sequence']} {J['nterm']}/{J['cterm']}/{J['lys']} {J['state']} "
            f"atoms={top.getNumAtoms()} target={J['ns']:g}ns chunk_ns={chunk_ns:g} dt={J['dt_fs']}fs "
            f"device={dev} host={socket.gethostname()}")
        st = equilibrate(sim, top, J["run_seed"])
        sim, top, dev = make_simulation(True, J["run_seed"] + 1)  # not the NVT stream again
        sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        sim.context.setPositions(st.getPositions())
        sim.context.setVelocities(st.getVelocities())
        sim.context.setTime(0)
        save_state(sim, 0)
        k = 0
        log("NVT done; state_0000 saved")
    else:  # a fresh noise seed per resume point, so a resumed run never replays an earlier stream
        sim, top, dev = make_simulation(True, J["run_seed"] * 1000 + k)
        with open(os.path.join(args.jobdir, f"state_{k:04d}.xml")) as f:
            sim.context.setState(XmlSerializer.deserialize(f.read()))
        log(f"RESUME {J['job']} at chunk {k} ({k * chunk_ns:g} ns) device={dev} host={socket.gethostname()}")

    pep = peptide_atoms(top)
    steps = int(round(chunk_ns * 1e6 / J["dt_fs"]))
    while k < n_chunks:
        if args.until_ns is not None and k * chunk_ns >= args.until_ns - 1e-9:
            log(f"HELD at {k * chunk_ns:g}/{J['ns']:g} ns (--until-ns {args.until_ns:g})")
            return
        xtc = os.path.join(traj, f"chunk_{k + 1:04d}.xtc")
        sim.reporters = [XTCReporter(xtc, FRAME_STEPS, atomSubset=pep)]
        sim.currentStep = 0  # frame 1 of every chunk is at 20 ps
        t0 = time.time()
        sim.step(FRAME_STEPS)  # the chunk's first frame doubles as a timing probe
        sim.context.getState(getEnergy=True)
        if t0 + 1.05 * (time.time() - t0) * steps / FRAME_STEPS > deadline:
            sim.reporters = []
            os.remove(xtc)  # the chunk would end after the deadline: drop its first frame and stop at state k
            log(f"PAUSED at {k * chunk_ns:g}/{J['ns']:g} ns (time limit)")
            return
        sim.step(steps - FRAME_STEPS)
        sim.reporters = []
        save_state(sim, k + 1)
        k += 1
        log(f"{k * chunk_ns:g}/{J['ns']:g} ns  {chunk_ns / (time.time() - t0) * 86400:.0f} ns/day  {dev}")
    log("DONE")


if __name__ == "__main__":
    bench() if args.bench else run()
