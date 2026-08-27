#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="${TMPDIR:-/tmp}/activation-function-tuning-test"
mkdir -p "${build_dir}"

mpicxx -std=c++20 -O0 -g -I"${repo_root}/CNN/src" \
    "${repo_root}/CNN/experiments/activation-function-tuning/main.cpp" \
    "${repo_root}"/CNN/src/core/*.cpp \
    "${repo_root}"/CNN/src/data/*.cpp \
    "${repo_root}"/CNN/src/layers/*.cpp \
    "${repo_root}"/CNN/src/model/*.cpp \
    "${repo_root}"/CNN/src/optimizers/*.cpp \
    "${repo_root}"/CNN/src/training/*.cpp \
    "${repo_root}"/CNN/src/tuning/*.cpp \
    -o "${build_dir}/activation-function-tuning"

help="$(${build_dir}/activation-function-tuning --help)"
[[ "${help}" == *"--diagnostic"* ]]
[[ "${help}" == *"6 activation candidates"* ]]
[[ "${help}" == *"0.01, 0.05, 0.1"* ]]

diagnostic_help="$(${build_dir}/activation-function-tuning --diagnostic --help)"
[[ "${diagnostic_help}" == "${help}" ]]
