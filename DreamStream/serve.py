#!/usr/bin/env python3
"""Launch one DreamStream batch server per GPU slot.

Server i listens on --base-port + i. Any extra arguments are forwarded to
every server (see ``python -m dreamstream.server --help``), e.g.

    python serve.py --num-gpus 2 -- --wan-root /path/to/Wan2.2-Fun-5B-Control
"""
import argparse
import os
import signal
import subprocess
import sys

procs: list[subprocess.Popen] = []
_shutting_down = False


def cleanup(signum, frame):
    global _shutting_down
    if _shutting_down:
        print("\nSecond interrupt — SIGKILL'ing all children and exiting.", flush=True)
        for p in procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        os._exit(130)
    _shutting_down = True

    print("\nShutting down all servers (press Ctrl+C again to force quit)...", flush=True)
    for p in procs:
        try:
            p.terminate()
        except ProcessLookupError:
            pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"Force killing PID {p.pid}", flush=True)
            try:
                p.kill()
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"PID {p.pid} unresponsive to SIGKILL (kernel D-state); abandoning.", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num-gpus", type=int, default=1,
                    help="Number of GPUs (one server per GPU)")
parser.add_argument("--servers-per-gpu", type=int, default=1,
                    help="Number of server instances per GPU")
parser.add_argument("--base-port", type=int, default=8790)
parser.add_argument("--host", default="0.0.0.0")
args, server_args = parser.parse_known_args()
if server_args[:1] == ["--"]:
    server_args = server_args[1:]

repo_dir = os.path.dirname(os.path.abspath(__file__))
server_id = 0
for gpu in range(args.num_gpus):
    for s in range(args.servers_per_gpu):
        port = args.base_port + server_id
        print(f"Starting DreamStream server on GPU {gpu}, port {port} "
              f"(instance {s+1}/{args.servers_per_gpu})")
        p = subprocess.Popen(
            [
                sys.executable, "-m", "dreamstream.server",
                "--host", args.host,
                "--port", str(port),
                *server_args,
            ],
            cwd=repo_dir,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
        procs.append(p)
        server_id += 1

print(f"Launched {server_id} DreamStream servers (PIDs: {[p.pid for p in procs]})")

for p in procs:
    p.wait()
