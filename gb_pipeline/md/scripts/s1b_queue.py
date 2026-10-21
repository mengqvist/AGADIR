"""Run a list of Stage 1b jobs on every GPU of this machine, one job per GPU at a time (N153).

Usage:
  python s1b_queue.py --root RUNS --jobs 4g_helix1,4h_helix1,... [--gpus 0,1]
                      (--stop-at-uk 21:00 | --max-hours 11.5)
  python s1b_queue.py --root RUNS --bench 30 [--jobs 4g_coil1] [--gpus 0,1]
RUNS holds one directory per job (built by s1b_prepare.py). Each GPU takes the next unfinished job from
the list when its current job ends; finished jobs (DONE) are skipped, unfinished ones resume. Every job is
run by s1b_run.py with one shared deadline, so a job pauses at a chunk boundary rather than being killed.
Local runs: --stop-at-uk 21:00 (the owner's limit; British time, BST/GMT handled). Cloud: --max-hours,
below the platform's session limit. A job must be listed on one machine at a time.
--bench MIN: s1b_run.py --bench on every GPU at once (same job), then a summary line per GPU.
"""
import argparse
import datetime
import os
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--jobs", default="4g_coil1")
ap.add_argument("--gpus", help="comma-separated device indices (default: all)")
ap.add_argument("--stop-at-uk", help="HH:MM British time today")
ap.add_argument("--max-hours", type=float)
ap.add_argument("--bench", type=float)
ap.add_argument("--platform", default="CUDA", help="passed to s1b_run.py (CPU only for tests)")
ap.add_argument("--until-ns", type=float, help="passed to s1b_run.py: stop each job (HELD) at this length")
ap.add_argument("--min-start-minutes", type=float, default=20.0,
                help="do not start a job (or begin a fresh one) with less time than this left")
args = ap.parse_args()


def gpus():
    if args.gpus:
        return args.gpus.split(",")
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU")] or ["0"]


def deadline():
    if args.stop_at_uk:
        hh, mm = map(int, args.stop_at_uk.split(":"))
        uk = ZoneInfo("Europe/London")
        now = datetime.datetime.now(uk)
        end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if end <= now:
            sys.exit(f"{args.stop_at_uk} UK time has passed today; nothing started")
        return end.timestamp()
    if args.max_hours:
        return time.time() + 3600 * args.max_hours
    sys.exit("give --stop-at-uk or --max-hours (no open-ended runs)")


def status(job):
    p = os.path.join(args.root, job, "progress.log")
    if not os.path.exists(p):
        return "new"
    last = open(p).read().rstrip().splitlines()[-1]
    return "done" if last.endswith("DONE") else "paused" if "PAUSED" in last else "held" if "HELD" in last else "partial"


def stream(cmd, tag):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if "Warning" not in line:
            print(f"[gpu{tag}] {line}", end="", flush=True)
    return proc.wait()


def main():
    devs = gpus()
    run = [sys.executable, os.path.join(HERE, "s1b_run.py"), "--platform", args.platform]
    if args.bench:
        job = os.path.join(args.root, args.jobs.split(",")[0])
        ts = [threading.Thread(target=stream, args=(run + [job, "--device", d, "--bench", str(args.bench)], d))
              for d in devs]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for line in open(os.path.join(job, "bench.log")):
            if "SUMMARY" in line:
                print(line, end="")
        return
    end = deadline()
    todo = [j for j in args.jobs.split(",") if j]
    missing = [j for j in todo if not os.path.exists(os.path.join(args.root, j, "job.json"))]
    if missing:
        sys.exit(f"not found under {args.root}: {missing}")
    todo = [j for j in todo if status(j) != "done"]
    lock = threading.Lock()
    print(f"{len(todo)} jobs on gpus {devs}, deadline {time.strftime('%Y-%m-%d %H:%M', time.gmtime(end))} UTC",
          flush=True)

    def worker(d):
        while True:
            if time.time() + 60 * args.min_start_minutes > end:
                print(f"[gpu{d}] less than {args.min_start_minutes:g} min left: stopping", flush=True)
                return
            with lock:
                if not todo:
                    return
                job = todo.pop(0)
            extra = ["--until-ns", str(args.until_ns)] if args.until_ns else []
            rc = stream(run + [os.path.join(args.root, job), "--device", d, "--stop-at", str(end)] + extra, d)
            if rc != 0:
                print(f"[gpu{d}] {job} exited with code {rc}; not retried in this session", flush=True)
            elif status(job) == "paused":
                return  # the deadline is near; later jobs would not start either

    ts = [threading.Thread(target=worker, args=(d,)) for d in devs]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("summary:", ", ".join(f"{j}={status(j)}" for j in args.jobs.split(",") if j), flush=True)


if __name__ == "__main__":
    main()
