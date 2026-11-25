#!/usr/bin/env python3
"""Sample server RSS and device memory during a benchmark session."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def proc_value(pid: int, key: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return 0


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def main() -> None:
    pid_file = Path(sys.argv[1])
    name = sys.argv[2]
    pid = int(pid_file.read_text())
    output = HERE / "raw" / f"memory-{name}.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("elapsed_s", "gpu_used_mib", "process_gpu_mib", "process_rss_mib", "process_hwm_mib", "mem_available_mib"))
        start = time.monotonic()
        while Path(f"/proc/{pid}").exists():
            try:
                gpu = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                apps = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
                gpu, apps = "0", []
            process_gpu = 0
            for row in apps:
                fields = [value.strip() for value in row.split(",")]
                if len(fields) == 2 and fields[0] == str(pid):
                    process_gpu = int(fields[1])
            writer.writerow(
                (
                    f"{time.monotonic() - start:.2f}",
                    gpu,
                    process_gpu,
                    proc_value(pid, "VmRSS") // (1024 * 1024),
                    proc_value(pid, "VmHWM") // (1024 * 1024),
                    mem_available() // (1024 * 1024),
                )
            )
            file.flush()
            time.sleep(1)
    print(output)


if __name__ == "__main__":
    main()
