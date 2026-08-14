#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TEST_FILE="${TEST_FILE:-test_cases/test.txt}"
RESULT_DIR="${RESULT_DIR:-results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NSYS_BIN="${NSYS_BIN:-nsys}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="${NCU_SET:-full}"

mkdir -p "$RESULT_DIR"

make_reference_submission() {
    local tmp_dir="$1"
    printf '%s\n' \
        'from reference import ref_kernel as custom_kernel' \
        > "$tmp_dir/submission.py"
}

target_python_path() {
    local target="$1"
    local tmp_dir="$2"
    local python_path="$PWD:$PWD/.."

    if [ "$target" = "reference" ]; then
        make_reference_submission "$tmp_dir"
        python_path="$tmp_dir:$python_path"
    fi

    printf '%s\n' "$python_path"
}

run_eval() {
    local target="$1"
    local mode="$2"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local python_path
    python_path="$(target_python_path "$target" "$tmp_dir")"
    local output_file="$RESULT_DIR/flashattention-${target}-${mode}.txt"
    local status=0

    echo "==> ${target} ${mode}: ${TEST_FILE}"
    set +e
    PYTHONPATH="$python_path" "$PYTHON_BIN" ../eval.py "$mode" "$TEST_FILE" | tee "$output_file"
    status="${PIPESTATUS[0]}"
    set -e

    rm -rf "$tmp_dir"

    return "$status"
}

run_nsys() {
    local target="$1"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local python_path
    python_path="$(target_python_path "$target" "$tmp_dir")"
    local prefix="$RESULT_DIR/flashattention-${target}-nsys"
    local profile_log="$RESULT_DIR/flashattention-${target}-nsys-profile.txt"
    local stats_file="$RESULT_DIR/flashattention-${target}-nsys-stats.txt"
    local status=0

    echo "==> ${target} nsys: ${TEST_FILE}"
    set +e
    PROFILE_TEST_FILE="$TEST_FILE" PYTHONPATH="$python_path" "$NSYS_BIN" profile \
        --trace=cuda,cublas,nvtx,osrt \
        --stats=false \
        --force-overwrite true \
        -o "$prefix" \
        "$PYTHON_BIN" -c \
        'import os, torch; from eval import get_test_cases; from reference import generate_input; from submission import custom_kernel; tests = get_test_cases(os.environ["PROFILE_TEST_FILE"], None); data = generate_input(**tests[-1].args); torch.cuda.synchronize(); custom_kernel(data); torch.cuda.synchronize()' \
        > "$profile_log" 2>&1
    status="$?"
    set -e

    if [ "$status" -eq 0 ]; then
        set +e
        "$NSYS_BIN" stats --force-export=true "${prefix}.nsys-rep" | tee "$stats_file"
        status="${PIPESTATUS[0]}"
        set -e
    fi

    rm -rf "$tmp_dir"
    return "$status"
}

run_ncu() {
    local target="$1"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local python_path
    python_path="$(target_python_path "$target" "$tmp_dir")"
    local prefix="$RESULT_DIR/flashattention-${target}-ncu"
    local output_file="$RESULT_DIR/flashattention-${target}-ncu.txt"
    local profile_log="$RESULT_DIR/flashattention-${target}-ncu-profile.txt"
    local status=0
    local kernel_args=()

    if [ "$target" = "submission" ]; then
        kernel_args=(
            --kernel-name-base demangled
            --kernel-name 'regex:.*_flash_attention_kernel.*'
        )
    fi

    echo "==> ${target} ncu: ${TEST_FILE}"
    set +e
    PROFILE_TEST_FILE="$TEST_FILE" PYTHONPATH="$python_path" "$NCU_BIN" \
        --target-processes all \
        --set "$NCU_SET" \
        --force-overwrite \
        --export "$prefix" \
        "${kernel_args[@]}" \
        "$PYTHON_BIN" -c \
        'import os, torch; from eval import get_test_cases; from reference import generate_input; from submission import custom_kernel; tests = get_test_cases(os.environ["PROFILE_TEST_FILE"], None); data = generate_input(**tests[-1].args); torch.cuda.synchronize(); custom_kernel(data); torch.cuda.synchronize()' \
        > "$profile_log" 2>&1
    status="$?"
    set -e

    if [ "$status" -eq 0 ]; then
        set +e
        "$NCU_BIN" --import "${prefix}.ncu-rep" --page details | tee "$output_file"
        status="${PIPESTATUS[0]}"
        set -e
    fi

    if [ "$status" -ne 0 ]; then
        echo "ncu failed; see $profile_log" >&2
    fi

    rm -rf "$tmp_dir"
    return "$status"
}

run_profile() {
    local target="$1"

    run_nsys "$target"
    run_ncu "$target"
}

if [ "$#" -eq 0 ]; then
    set -- submission-test submission-benchmark reference-test reference-benchmark
fi

for item in "$@"; do
    case "$item" in
        test|benchmark|profile)
            if [ "$item" = "profile" ]; then
                run_profile submission
            else
                run_eval submission "$item"
            fi
            ;;
        submission-test|submission-benchmark|submission-profile)
            if [ "$item" = "submission-profile" ]; then
                run_profile submission
            else
                run_eval submission "${item#submission-}"
            fi
            ;;
        reference-test|reference-benchmark|reference-profile)
            if [ "$item" = "reference-profile" ]; then
                run_profile reference
            else
                run_eval reference "${item#reference-}"
            fi
            ;;
        submission-nsys|reference-nsys)
            run_nsys "${item%-nsys}"
            ;;
        submission-ncu|reference-ncu)
            run_ncu "${item%-ncu}"
            ;;
        *)
            echo "unknown mode: $item" >&2
            echo "usage: $0 [test] [benchmark] [profile] [submission-nsys] [submission-ncu] [reference-profile]" >&2
            exit 2
            ;;
    esac
done
