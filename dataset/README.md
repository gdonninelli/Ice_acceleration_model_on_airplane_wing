# Dataset

`cnn_dataset_train.npz` and `cnn_dataset_test.npz` are **not tracked by git**
(`.gitignore:6`). Regenerate them from the committed inputs with the commands
below; `CHECKSUMS.txt` records the SHA-256 of the accepted build so you can
confirm you reproduced it exactly.

## Why they are not committed

The train file is 128 MB, above GitHub's 100 MB per-file limit, and the pair
totals 160 MB. Committing them needs Git LFS, which is not configured in this
repository — that is a decision for the group, not a side effect of a
regeneration. See the caveat at the bottom.

## Regenerate

From the repository root, with an MPI toolchain and Python 3 + NumPy + pandas:

```bash
# 1. rotated profiles: 5 base .dat x 157 angles -> 785 files in SDF/data/
python3 SDF/data/rotate_profiles.py

# 2. signed distance fields: one _matrix.txt per .dat (790 total)
cd SDF && mpic++ main.cpp SDFGenerator.cpp -o sdfgen -std=c++17
mpirun -np 16 --oversubscribe ./sdfgen && cd ..

# 3. assemble the .npz (prints how many rows it keeps and discards)
python3 build_dataset.py --seed 42

# 4. verify before using
python3 scripts/verify_dataset.py dataset/cnn_dataset_train.npz \
    --paired dataset/cnn_dataset_test.npz
python3 results/data_audit_v2/audit_v2.py
```

Expected counts at each stage:

| stage | count |
|---|---|
| base profiles | 5 |
| unique angles in the summaries | 157 |
| rotated `.dat` | 785 (+5 base = 790) |
| `_matrix.txt` | 790 |
| summary rows read | 2384 |
| discarded, \|angle\| > 14 | 242 |
| samples kept | **2142** |
| train / test (seed 42, 80/20) | 1713 / 429 |

Wall time on 20 cores: ~15 s for the SDF fields, ~5 s to assemble.

## Seeds

- `build_dataset.py --seed 42` drives the train/test split. It is the only
  random choice in the pipeline; every other stage is deterministic.
- Step 2 sorts its inputs, and step 3 sorts the summaries, so the assembly
  order does not depend on the filesystem.

## Schema

| array | shape | meaning |
|---|---|---|
| `X_sdf` | `(N, 150, 150)` | signed distance field; negative inside the profile. Grid spans x ∈ [-0.2, 1.1], z ∈ [-0.4, 0.4], so cells are **not square** (dx = 0.008725, dz = 0.005369) |
| `X_scalars` | `(N, 2)` | `[Reynolds, AoA in degrees]` |
| `Y_cl` | `(N,)` | lift coefficient, mean over the last 10% of the CFD run |
| `metadata` | `(N, 1)` | NACA profile id |

The angle of attack is encoded **twice**: in `X_scalars[:, 1]` and in the
geometry itself, since the profile is rotated before the field is computed.

## Accepted build

`CHECKSUMS.txt` corresponds to a build that passes every acceptance check:

- lag-1 autocorrelation 0.9998 / 0.9992 (a signed distance function is
  1-Lipschitz, hence smooth);
- negative-cell fraction 0.0756;
- |∇| coefficient of variation 0.131 (|∇SDF| is nearly constant);
- 1713/1713 targets found in the summaries;
- both scalar columns vary;
- angle reconstructed from the geometry agrees with the `angle` column to a
  median 0.127°, 100% within 0.5°;
- C_l vs α fits a slope of 0.1111 per degree against the 0.1097 predicted by
  thin-airfoil theory (1.3% agreement), intercept 0.0004, R² = 0.980.

## Caveats

1. **Untracked intermediates hid a failure for weeks.** `SDF/data/*_matrix.txt`
   and `SDF/data/*_alpha_*.dat` are gitignored, so the dataset that was in use
   could not be reproduced or inspected by anyone else — and it turned out to
   be Gaussian noise rather than distance fields. The `.gitignore` entries have
   been left untouched; whether to track these artifacts, adopt Git LFS, or
   publish a checksum-verified release is a decision for the group.
2. **The split is random, not grouped.** One geometry serves 4–5 Reynolds
   numbers, so a uniform split puts the same field on both sides: 303 of the
   429 test fields also appear in train. The test set therefore measures
   generalization to unseen (geometry, Reynolds) pairs, not to unseen
   geometries. A group-wise split on (profile, angle) would be needed for the
   stronger claim. Not changed here because it alters selection semantics.
3. **Angles beyond ±10° are past stall**, where C_l ≈ 2πα no longer holds and
   the SIMM physics term is not a valid prior. The current filter keeps
   \|α\| ≤ 14.
