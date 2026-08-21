# Top-level build orchestration for the ice acceleration model.
#
# The CNN subproject is built through its CMake project (which also compiles
# every experiment under CNN/experiments/*/), the SDF generator is a direct
# MPI compile. Run `make help` for the target list.

BUILD_DIR  ?= build/CNN
BUILD_TYPE ?= Release
NP         ?= 4
ARGS       ?=
MPICXX     ?= mpic++
PYTHON     ?= python3

SDF_BIN    := build/sdf/sdfgen

.PHONY: all cnn sdf experiments test run cross-validate dataset clean help

all: cnn sdf ## Build the CNN (with tests and experiments) and the SDF generator

cnn: ## Configure and build the CNN executable, tests, and experiments
	cmake -S CNN -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

experiments: cnn ## Alias: experiment binaries are built with the CNN into build/CNN/experiments

sdf: $(SDF_BIN) ## Build the SDF generator

$(SDF_BIN): SDF/main.cpp SDF/SDFGenerator.cpp SDF/SDFGenerator.hpp
	mkdir -p build/sdf
	$(MPICXX) SDF/main.cpp SDF/SDFGenerator.cpp -o $(SDF_BIN) -std=c++17

test: cnn ## Run the CTest suite (serial and 2-rank MPI)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: cnn ## Train the CNN: make run NP=4 ARGS="--epochs 100 --dropout 0.2"
	mpirun -n $(NP) $(BUILD_DIR)/cnn_executable $(ARGS)

cross-validate: cnn ## K-fold CV: make cross-validate NP=4 ARGS="--folds 5"
	mpirun -n $(NP) $(BUILD_DIR)/cnn_executable --cross-validate $(ARGS)

dataset: sdf ## Regenerate dataset/*.npz (needs the data-pipeline portability fix; see dataset docs)
	@grep -q -- "--seed" build_dataset.py || { \
	  echo "ERROR: build_dataset.py has no --seed option, so this checkout lacks the"; \
	  echo "data-pipeline portability fix and would assemble a non-reproducible (or"; \
	  echo "invalid) dataset. Merge fix/data-pipeline-portability first, or regenerate"; \
	  echo "from a worktree of that branch (see dataset/README.md there)."; \
	  exit 1; }
	$(PYTHON) SDF/data/rotate_profiles.py
	cd SDF && mpirun -np $(NP) --oversubscribe ../$(SDF_BIN)
	$(PYTHON) build_dataset.py --seed 42

clean: ## Remove every build artifact
	rm -rf build

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
