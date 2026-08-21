# Pull Request History

This logbook records completed pull requests so contributors can see when a
change was introduced, who implemented it, and how it was validated. Add one
entry per pull request, newest first.

## Template

### PR: <title or number>

- **Status:** Merged
- **Date:** YYYY-MM-DD
- **Author:** Name (`email`)
- **Branch:** `feature/branch-name`
- **Commit(s):** `<short-sha>`
- **Summary:** Short description of the delivered feature.
- **Implementation:** Main technical decisions and affected components.
- **Validation:** Builds, tests, reviews, or manual checks completed.

## Dropout, Makefile, and physics-weight tuning

- **Status:** Pending merge
- **Implementation dates:** 2026-08-16 to 2026-08-21
- **Review date:** 2026-08-21
- **Authors:** Vittorio Sironi (`vittorio.sironi@icloud.com`) - `51820f9`,
  `6ef94e9`, `ca9fdaa`, `f4ce5f9`, `fb1fc2a`, `28d39eb`; Alessia Rigoni
  (`a.rigoni5@campus.unimib.it`) - `d4fc0ff`
- **Branch:** `feature/dropout-makefile-physics-tuning`
- **Base:** `main`
- **Reviewed head:** `d4fc0ff2e1680c82fa7e1e5626f026744b8bbe9e`
- **PR:** https://github.com/gdonninelli/Ice_acceleration_model_on_airplane_wing/pull/9
- **Summary:** Added deterministic MPI-rank-invariant inverted dropout, a
  top-level Makefile/CMake experiment build path, and paired dropout-rate and
  SIMM physics-weight tuning experiments with committed artifacts.
- **Implementation:**
  - Added `LayerExecutionContext`, `DropoutLayer`, `Recipes::dropout`, and the
    `--dropout` CLI option; the trainer supplies per-batch deterministic context
    and prediction restores inference behavior.
  - Added Makefile targets and automatic experiment executable discovery in
    CMake; `make dataset` explicitly blocks on the missing seed-capable data
    pipeline.
  - Added two 5-fold paired tuning executables, analyzers, documentation, and
    per-fold CSV/diagnostic artifacts.
- **Validation:**
  - `make test` built all CNN and experiment targets; serial and MPI-2 CTest
    suites passed.
  - `git diff --check`, analyzer Python compilation, both committed sweep
    analyzers, CLI/help smoke tests, and representative metadata JSON parsing
    passed.
  - Full 100-epoch sweeps were not rerun in this review; `make dataset` was
    confirmed to stop at its documented portability guard.
- **Per-PR report:** `PR_REVIEWS/feature-dropout-makefile-physics-tuning.md`
- **Review outcome:** No critical findings; the medium dropout shape-validation
  finding and incomplete result documentation were resolved with regression
  tests and numerical README analysis. Low analyzer/metadata and provenance
  gaps remain non-blocking.

## Implementing Internal Weights Analysis

- **Status:** Pending merge
- **Review date:** 2026-08-07
- **Implementation date:** 2026-08-07
- **Author:** Giulio Donninelli (`hello@giuliodonninelli.it`)
- **Branch:** `feature/store-gradient`
- **Base:** `main`
- **Reviewed head:** `1163eea41ffac35863380fbd40168195ba3be65a`
- **Commit(s):** `1163eea`
- **Summary:** Added optional MPI-aware training diagnostics, cross-validation artifacts, plotting support, CLI controls, and CMake/CTest integration.
- **Implementation:**
  - Added model layer/parameter metadata and activation observation during the existing training forward pass.
  - Added gradient, update-ratio, activation, learning-rate, epoch, and metadata artifact recording with MPI aggregation and rank-zero output.
  - Added candidate/fold diagnostic context, cross-validation summaries, plotting utilities, documentation, and tests.
- **Validation:** CMake build, serial and MPI-2 CTest, direct MPI-2 test execution, `git diff --check`, Python compilation, and a warning-enabled compile completed successfully; dataset-backed training and plotting smoke tests were unavailable because required NPZ/artifact inputs are absent.
- **Per-PR report:** `PR_REVIEWS/feature-store-gradient.md`
- **Review outcome:** No critical findings; one compiler warning and dataset-dependent test gaps remain non-blocking.
## Feature/l1 l2 regularization

- **Status:** Pending merge
- **Implementation dates:** 2026-08-06 to 2026-08-07
- **Review date:** 2026-08-08
- **Author:** Alessandro Ferdinando Verrengia (`alessandro.verrengia@mail.polimi.it`)
- **Branch:** `feature/l1-l2-regularization`
- **Base:** `main`
- **Reviewed head:** `1e097aa1c1d34280945bdcc09346eee169298cf3`
- **PR:** https://github.com/gdonninelli/Ice_acceleration_model_on_airplane_wing/pull/7
- **Commit(s):** `0c0d7f8`, `6210681`, `5128a24`, `96e97e1`, `303bdee`, `9edc8e7`, `1e097aa`
- **Summary:** Added weight-only L1/L2 regularization to CNN training and a paired cross-validation experiment for selecting penalty strengths.
- **Implementation:**
  - Added L1/L2 configuration and CLI options, with distributed gradient penalties applied after data-gradient synchronization.
  - Added standalone L1/L2 sweeps, CSV recording, paired analysis, and committed experiment results/documentation.
- **Validation:**
  - Production and experiment sources compiled warning-free with `mpicxx -std=c++20 -O3 -Wall -Wextra -Wpedantic`.
  - Cross-validation tests, including invalid L1/L2 configuration guards, passed serially and with two MPI ranks; both committed CSVs analyzed to lambda = 0.
  - Two-rank one-epoch real-dataset smoke training with non-zero L1/L2 completed; `git diff --check` passed.
- **Follow-up:** Added `Trainer` API-boundary validation for finite non-negative L1/L2 values and corrected README dataset, runtime, and result claims.
- **Review report:** `PR_REVIEWS/feature-l1-l2-regularization.md`
- **Review outcome:** No critical findings; previously reported medium API-validation and low documentation findings are resolved in the follow-up changes.

## Activation Functions

- **Status:** Pending merge
- **Date:** 2026-07-17
- **Author:** Alessandro Ferdinando Verrengia (`alessandro.verrengia@mail.polimi.it`)
- **Branch:** `feature/activation-functions`
- **Commit(s):** `eb97d5c` - Add reusable activation layers and refactor Leaky ReLU
- **Summary:** Added runtime selection of Leaky ReLU, ReLU, Tanh, and Sigmoid
  activations for the CNN. The default remains Leaky ReLU with `alpha = 0.05`.
- **Implementation:**
  - Added `ActivationLayer` to share element-wise forward and backward logic.
  - Added concrete ReLU, Tanh, and Sigmoid layers; refactored Leaky ReLU to use
    the common base class.
  - Added `make_activation(name, alpha)` to create independent, case-insensitive
    activation instances for each network position.
  - Added `--activation` and `--alpha` command-line options and README usage.
  - Added review follow-up validation that requires `--alpha` to be finite and
    non-negative before constructing the model.
- **Validation:**
  - Full CNN build completed with `mpicxx`.
  - A focused activation smoke test covered all factory choices, forward and
    backward execution, and case-insensitive lookup.
  - Invalid activation names, malformed alpha values, negative alpha values,
    and `nan` alpha values fail with an error.
  - `git diff --check` completed without whitespace errors.
