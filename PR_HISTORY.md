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
