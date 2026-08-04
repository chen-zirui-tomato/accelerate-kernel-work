#!/usr/bin/env bash
set -euo pipefail

# Measure the official reference implementation and/or your submission for all
# assignment problems. Run this script from the repository root.
#
# Usage:
#   ./measure_all.sh reference    # only reference.py
#   ./measure_all.sh submission   # only submission.py
#   ./measure_all.sh both         # default
#
# Outputs go to results/, using names like:
#   swiglu-reference-benchmark.txt
#   swiglu-reference-nsys-stats.txt
#   swiglu-submission-test.txt
#   swiglu-submission-benchmark.txt
#   swiglu-submission-nsys-stats.txt

MODE="${1:-both}"
if [[ "$MODE" != "reference" && "$MODE" != "submission" && "$MODE" != "both" ]]; then
  echo "Usage: $0 [reference|submission|both]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$ROOT/results"
RUNNER="${TMPDIR:-/tmp}/asst5_measure_runner.py"
PROBLEMS=(
  "histogram"
  "rk4"
  "1d-occupancy-decoder"
  "flashattention"
  "swiglu"
)

mkdir -p "$RESULTS"

cat > "$RUNNER" <<'PY'
import dataclasses
import gc
import math
import os
import re
import sys
import time
from pathlib import Path

import torch


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def parse_test_cases(path):
    match = r"\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*|[+-]?[0-9]+)\s*"
    tests = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        case = {}
        for part in line.split(";"):
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                raise ValueError(f"invalid test case: {line!r}: {part!r}")
            value = matched[2]
            try:
                value = int(value)
            except ValueError:
                pass
            case[matched[1]] = value
        tests.append((line, case))
    return tests


def calculate_stats(durations):
    runs = len(durations)
    mean = sum(durations) / runs
    if runs > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in durations) / (runs - 1))
    else:
        std = 0.0
    err = std / math.sqrt(runs)
    return Stats(runs, mean, std, err, min(durations), max(durations))


def clone_data(data):
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    if isinstance(data, list):
        return [clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    return data


def output_summary(value):
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)} {value.dtype} {value.device}"
    if isinstance(value, tuple):
        return "(" + ", ".join(output_summary(x) for x in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(output_summary(x) for x in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {output_summary(v)}" for k, v in value.items()) + "}"
    return type(value).__name__


def load_impl(kind):
    from reference import generate_input
    if kind == "reference":
        from reference import ref_kernel as kernel
        checker = None
    elif kind == "submission":
        from reference import check_implementation
        from submission import custom_kernel as kernel
        checker = check_implementation
    else:
        raise ValueError(f"unknown implementation kind: {kind}")
    return generate_input, kernel, checker


def run_benchmark(kind, test_file):
    generate_input, kernel, checker = load_impl(kind)
    tests = parse_test_cases(test_file)
    print(f"benchmark-count: {len(tests)}")

    for idx, (spec, args) in enumerate(tests):
        print(f"benchmark.{idx}.spec: {spec}", flush=True)
        data = generate_input(**args)

        if checker is not None:
            check_copy = clone_data(data)
            torch.cuda.synchronize()
            output = kernel(data)
            torch.cuda.synchronize()
            good, message = checker(check_copy, output)
            if not good:
                print(f"benchmark.{idx}.status: fail")
                print(f"benchmark.{idx}.error: {message}")
                continue

        for _ in range(3):
            torch.cuda.synchronize()
            output = kernel(data)
            torch.cuda.synchronize()
            del output

        durations = []
        wall_start = time.perf_counter_ns()
        for i in range(100):
            gc.collect()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = kernel(data)
            end.record()
            torch.cuda.synchronize()
            durations.append(start.elapsed_time(end))
            del output

            if i >= 2:
                stats = calculate_stats(durations)
                wall_elapsed = time.perf_counter_ns() - wall_start
                if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > 10_000 or wall_elapsed > 120e9:
                    break

        stats = calculate_stats(durations)
        print(
            f"benchmark.{idx}: runs={stats.runs}, mean={stats.mean:,.3f} ms, "
            f"std={stats.std:,.3f} ms, err={stats.err:,.3f} ms, "
            f"best={stats.best:,.3f} ms, worst={stats.worst:,.3f} ms",
            flush=True,
        )
    print("check: pass")


def run_profile(kind, test_file):
    generate_input, kernel, checker = load_impl(kind)
    spec, args = parse_test_cases(test_file)[-1]
    print(f"profile.spec: {spec}", flush=True)
    data = generate_input(**args)
    if checker is not None:
        check_copy = clone_data(data)
        torch.cuda.synchronize()
        output = kernel(data)
        torch.cuda.synchronize()
        good, message = checker(check_copy, output)
        if not good:
            raise RuntimeError(message)
        del output

    torch.cuda.synchronize()
    output = kernel(data)
    torch.cuda.synchronize()
    print(f"profile.output: {output_summary(output)}", flush=True)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"benchmark", "profile"}:
        print("usage: _measure_runner.py <benchmark|profile> <reference|submission>", file=sys.stderr)
        return 2
    command, kind = sys.argv[1], sys.argv[2]
    test_file = "test_cases/test.txt"
    if command == "benchmark":
        run_benchmark(kind, test_file)
    else:
        run_profile(kind, test_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

run_reference() {
  local problem="$1"
  echo "==== $problem reference benchmark ===="
  (
    cd "$ROOT/problems/$problem"
    PYTHONPATH="$PWD:$PWD/.." python "$RUNNER" benchmark reference \
      | tee "$RESULTS/$problem-reference-benchmark.txt"
  )

  echo "==== $problem reference nsys ===="
  (
    cd "$ROOT/problems/$problem"
    PYTHONPATH="$PWD:$PWD/.." nsys profile \
      --trace=cuda,cublas,nvtx,osrt \
      --stats=true \
      --force-overwrite true \
      -o "$RESULTS/$problem-reference-nsys" \
      python "$RUNNER" profile reference

    nsys stats --force-export=true "$RESULTS/$problem-reference-nsys.nsys-rep" \
      | tee "$RESULTS/$problem-reference-nsys-stats.txt"
  )
}

run_submission() {
  local problem="$1"
  if [[ ! -f "$ROOT/problems/$problem/submission.py" ]]; then
    echo "==== $problem submission skipped: submission.py not found ===="
    return 0
  fi

  echo "==== $problem submission test ===="
  (
    cd "$ROOT/problems/$problem"
    PYTHONPATH="$PWD:$PWD/.." python ../eval.py test test_cases/test.txt \
      | tee "$RESULTS/$problem-submission-test.txt"
  )

  echo "==== $problem submission benchmark ===="
  (
    cd "$ROOT/problems/$problem"
    PYTHONPATH="$PWD:$PWD/.." python "$RUNNER" benchmark submission \
      | tee "$RESULTS/$problem-submission-benchmark.txt"
  )

  echo "==== $problem submission nsys ===="
  (
    cd "$ROOT/problems/$problem"
    PYTHONPATH="$PWD:$PWD/.." nsys profile \
      --trace=cuda,cublas,nvtx,osrt \
      --stats=true \
      --force-overwrite true \
      -o "$RESULTS/$problem-submission-nsys" \
      python "$RUNNER" profile submission

    nsys stats --force-export=true "$RESULTS/$problem-submission-nsys.nsys-rep" \
      | tee "$RESULTS/$problem-submission-nsys-stats.txt"
  )
}

for problem in "${PROBLEMS[@]}"; do
  if [[ "$MODE" == "reference" || "$MODE" == "both" ]]; then
    run_reference "$problem"
  fi

  if [[ "$MODE" == "submission" || "$MODE" == "both" ]]; then
    run_submission "$problem"
  fi
done

echo "All requested measurements are in $RESULTS"
