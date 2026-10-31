# Random seeds used under sim/a5 (all >= 200000; 12345 and 24680 are never used)

| seed | where |
|---|---|
| 200001 | tests/test_tasks_mul.py reference vs integer product |
| 200002 | tests/test_tasks_mul.py random generators (SEED + 1) |
| 200011 | tests/test_oracle.py random tables vs TernaryNet, 1-D (SEED 200010 + dims) |
| 200012 | tests/test_oracle.py random tables vs TernaryNet, 2-D |
| 200020 | tests/test_oracle.py random n=5,6 oracle samples (SEED + 10) |
| 200021 | tests/test_k2_search.py monotone-bound test |
| 200022 | tests/test_k2_search.py prefix-soundness test |
| 200101 | k2_search.py anneal default (--seed) |
| 200201 | delay.py verify_construction random words (+ d) |
| 200301 | tests/test_m1d_cert.py random tables (K=2,3,4) |
| 200302 | tests/test_m1d_cert.py random tables for the accuracy bound (SEED + 1 / + 2 = 200302 / 200303) |
| 200401 | tests/test_delay.py (+1 = 200402 pigeonhole random tables) |
| 200501 | tests/test_oracle_k4.py random n=5..8 (+1 = 200502 K=16) |
| 200601 | wave_cert.py find_counterexample random grids; tests/test_wave_cert.py relabelling (+1 = 200602 random bit flips) |
| 200701 | tests/test_k2_sat.py random flips (+1 = 200702, +2 = 200703); k2_sat.py report_solution delay generalisation words |
| 200702+n | k2_sat.py _random_m2d generalisation grids (seed 200702 + n, n = 6..8) |
