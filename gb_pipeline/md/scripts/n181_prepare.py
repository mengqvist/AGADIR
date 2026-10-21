"""Build the N181 coil-ensemble jobs: Ac-A16-NH2, unrestrained coil replicas, for s1b_run.py / s1b_queue.py.

Usage: ../.venv/bin/python n181_prepare.py [-j WORKERS] [--ns NS] [JOB ...]   (default: cA16_coil1..8)
Writes runs/n181/<job>/ exactly as s1b_prepare.py writes runs/s1b/<job>/ (system.xml.gz, start.pdb, peptide.pdb,
job.json), results/n181_build.txt and runs/n181/jobs.tsv.

Purpose (N181 step 2): backbone geometry of a poly-Ala peptide coil. GB pair energies are recomputed on its
non-helical frames with side chains threaded on (gb_pipeline/postprocessing); MD sets no constant (N148).
Setup = the admitted Stage 1b protocol (N153, N149 rev 4), unchanged: CHARMM36m (charmm36_2024.xml) + CHARMM TIP3P,
Ac (ACE patch, merged into residue 1) and amide (CT2, HT1 cis to O, merged into residue 16), coil start from PPII/beta
draws (span <= 5.0 nm, clash-free), rhombic dodecahedron d = 6.5 nm, no ions (neutral peptide; Cl- only rule holds
trivially), PME 1.2 nm (switch 1.0), HBonds, HMR 3 amu, 4 fs, Langevin middle 1/ps, 278 K, MC barostat 1 bar after
1 ns NVT, peptide XTC every 20 ps. build.py is not edited: the host is added to its tables at run time here.
"""
import gzip
import json
import multiprocessing
import os
import sys
import zlib

import numpy as np
from openmm import NonbondedForce, XmlSerializer, unit
from openmm.app import PME, HBonds, PDBFile

import build

build.S1B_PEPTIDES["A16"] = "A" * 16
build.S1B_VARIANTS["cA16"] = ("A16", "ACE", "LYS")  # Lys form irrelevant: no Lys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs", "n181")
DEFAULT_JOBS = [f"cA16_coil{r}" for r in range(1, 9)]
NS, DISCARD_NS = 50.0, 10.0


def prepare(arg):
    job, ns = arg
    variant, rep = job.split("_")
    r = int(rep[len("coil"):])
    top, pos, forcefield, info = build.s1b_build(variant, "coil", r)
    system = forcefield.createSystem(top, nonbondedMethod=PME, nonbondedCutoff=1.2 * unit.nanometer,
                                     switchDistance=1.0 * unit.nanometer, constraints=HBonds,
                                     hydrogenMass=3 * unit.amu)
    templates = build.matched_templates(forcefield, top)
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                  for i in range(system.getNumParticles())])
    pep = [a.index for a in top.atoms() if a.residue.chain.index == 0]
    res_q = {rr.index + 1: round(float(sum(q[a.index] for a in rr.atoms())), 3)
             for rr in top.residues() if rr.chain.index == 0}
    n_ion = sum(1 for rr in top.residues() if rr.name in ("CL", "NA"))
    n_wat = sum(1 for rr in top.residues() if rr.name == "HOH")
    assert abs(q.sum()) < 1e-3 and n_ion == 0 and abs(q[pep].sum()) < 1e-3, (q.sum(), n_ion)
    assert templates[0][1].startswith("ALA-ACE") and templates[-1][1].startswith("ALA-CT2"), templates
    out = os.path.join(RUNS, job)
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "system.xml.gz"), "wt") as f:
        f.write(XmlSerializer.serialize(system))
    PDBFile.writeFile(top, pos, open(os.path.join(out, "start.pdb"), "w"), keepIds=True)
    sub = build.Modeller(top, pos)
    sub.delete([rr for rr in top.residues() if rr.chain.index != 0])
    PDBFile.writeFile(sub.topology, sub.positions, open(os.path.join(out, "peptide.pdb"), "w"), keepIds=True)
    info.update({
        "job": job, "node": "N181", "replica": r, "ns": ns, "discard_ns": DISCARD_NS, "frame_ps": 20, "dt_fs": 4,
        "temperature_K": 278.0, "pressure_bar": 1.0, "friction_per_ps": 1.0, "nvt_ns": 1.0,
        "run_seed": zlib.crc32(job.encode()) % 100000, "restrained_residues": None, "restraint_k": None,
        "n_restraint_torsions": 0, "atoms": top.getNumAtoms(), "peptide_atoms": len(pep), "waters": n_wat, "cl": 0,
        "peptide_charge": round(float(q[pep].sum()), 3),
        "residue_charges": {k: v for k, v in res_q.items() if abs(v) > 1e-6},
        "templates": [t for _, t in templates], "openmm_ff": list(build.FORCEFIELDS["c36m"]),
    })
    json.dump(info, open(os.path.join(out, "job.json"), "w"), indent=1)
    return info


def main():
    argv = sys.argv[1:]
    workers, ns = 1, NS
    while argv[:1] and argv[0].startswith("-"):
        if argv[0] == "-j":
            workers, argv = int(argv[1]), argv[2:]
        elif argv[0] == "--ns":
            ns, argv = float(argv[1]), argv[2:]
        else:
            sys.exit(f"unknown option {argv[0]}")
    jobs = [j for j in (argv or DEFAULT_JOBS) if not os.path.exists(os.path.join(RUNS, j, "job.json"))]
    os.makedirs(RUNS, exist_ok=True)
    log = open(os.path.join(ROOT, "results", "n181_build.txt"), "a")
    with multiprocessing.Pool(workers) as pool:
        for i in pool.imap(prepare, [(j, ns) for j in jobs]):
            t = i["templates"]
            line = (f"{i['job']}\tatoms={i['atoms']}\twater={i['waters']}\tions=0\tpeptide_q={i['peptide_charge']:+g}\t"
                    f"draws={i['draws']}\tN:{t[0]} C:{t[-1]}\tseq={i['sequence']}\tns={i['ns']:g}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    with open(os.path.join(RUNS, "jobs.tsv"), "w") as f:
        f.write("job\tvariant\tstate\treplica\tns\tatoms\n")
        for job in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, job, "job.json")
            if os.path.exists(p):
                j = json.load(open(p))
                f.write(f"{job}\t{j['variant']}\t{j['state']}\t{j['replica']}\t{j['ns']:g}\t{j['atoms']}\n")


if __name__ == "__main__":
    main()
