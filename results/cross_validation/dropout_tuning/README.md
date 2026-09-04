# Dropout Tuning Result Provenance

The current experiment driver in
`CNN/experiments/dropout_tuning/main.cpp` uses the topology selected by the
layer-architecture search:

```text
Conv2D(8, 5x5, stride 5) -> LeakyReLU(0.05) -> Flatten ->
Dense(1024) -> LeakyReLU -> Dense(512) -> LeakyReLU ->
Dense(256) -> LeakyReLU -> Dense(128) -> LeakyReLU -> Dense(1)
```

The committed results below this directory were generated before this source
correction. The historical sweep therefore used the smaller
`Dense(128) -> Dense(64) -> Dense(1)` head. This applies to
`sweep_dropout.csv` and to the diagnostic files under `sweep/`.

The existing numerical results have not been replaced or presented as results
for the corrected topology. In particular, their validation MSE values and
diagnostics must be interpreted as historical measurements of the smaller
network. A new five-fold run is required before making a numerical claim about
dropout on the winning topology. The source revision recorded by the old
diagnostics is `unknown`.
