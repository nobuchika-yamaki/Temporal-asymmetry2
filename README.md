# Temporal-asymmetry2
Temporal Dissipation Improves Computational Stability in Recurrent Neural Networks
Reproducibility Code for All Experiments
This repository contains the full simulation code used in the study examining how temporal asymmetry (dissipation) improves robustness in recurrent neural computation.
All figures and quantitative results in the manuscript can be reproduced by running the provided Python script.
1. File structure
temporal_dissipation_full_simulation.py   # Main simulation script (all tasks)
2. Requirements
Python ≥ 3.9
Required libraries:
numpy
matplotlib
scikit-learn
Install with:
pip install numpy matplotlib scikit-learn
How to execute
python temporal_dissipation_full_simulation.py
This will:
Run Task 1: Attractor-memory retrieval
Run Task 2: Sequence-tracking test
Run Task 3: Input–output mapping
Print all performance metrics
Generate all figures used in the manuscript
4. Summary of implemented tasks
Task 1 — Attractor memory retrieval
Hopfield-type recurrent matrix storing 3 binary patterns
Initial-state noise: σ = 0.0, 0.4, 1.2
Comparison of reversible (γ=0) vs dissipative (γ=0.2) dynamics
Outputs: retrieval accuracy for each noise level
Task 2 — Sequence tracking (0→1→2 transition)
External input switches every 6 seconds
Noise injected during dynamics
Measures ability to follow the correct pattern
Outputs: tracking accuracy across noise conditions
Task 3 — Input–output mapping (Reservoir computing)
Random reservoir (g = 0.9)
Train linear readout on f(x) = sin(πx1)cos(πx2)
Test under noise σ = 0.0–1.0
Outputs: mean squared error for each condition
5. Reproducibility notes
Numerical integration uses explicit Euler, dt = 0.01–0.02
γ parameters match the manuscript exactly
All random seeds are initialized for consistency
Running the script produces the same numerical values reported in the paper

