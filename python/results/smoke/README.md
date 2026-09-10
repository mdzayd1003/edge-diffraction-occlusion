# Smoke output

Produced by `make quick`. Unlike the other projects in this repository, these
numbers are **not** placeholders: the physics runs in seconds on any laptop, so
`spectrum.md`, `study.csv` and the figures are real output of the real model.

The one thing that is stand-in data is the listening test. `validation.md` is
computed from `data/listening_test.csv`, which was **generated**, not collected —
its provenance sidecar says so and the report repeats it. The correlation there
says the analysis path works; it says nothing about whether the model predicts
what people hear.

- `spectrum.md` / `.json` / `.png` — third-octave insertion loss for a 3 m
  barrier at 10 m, with the Maekawa chart plotted alongside for comparison
- `impulse_response.npy` / `.png` — the 48 kHz time-domain response
- `study.csv` / `.md` / `.json` — all 126 configurations
- `surface_thin_screen.png`, `sensitivity.png` — the marginal effects
- `validation.md` / `.json`, `listening.png` — the listening-test analysis, on
  simulated ratings

`../../data/matlab_reference_study.csv` is the same 126-configuration study run
through the MATLAB port. A Python test compares the two and fails if they drift
apart by more than 0.05 dB in any band; the current disagreement is 0.002 dB.
