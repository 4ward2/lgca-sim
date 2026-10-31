# Random seeds used by the A5 training code (all >= 300000 except the three deliberate Stage A replays; 12345 and 24680 never used)

| seed | where |
|---|---|
| 300001 + s (s = 0..4, 0..3, 7, 9), 300100 + s, 300008, 300010, 300020 | tests/test_train_rounding.py (BPNet init and random batches) |
| 300555, 300556, 300901 | tests/test_train_b.py (oracle evaluation sets, smoke run) |
| 300301 | tests/test_train_probe_dag.py (probe resume test) |
| 300777 | train_b.py dev set (size-keyed sub-streams `[seed, n, 17]`) |
| 300778 | train_b.py test set |
| 300101-300103 | jobs_a5.json B pilot seeds (BPNet init, training stream `seed+1000`, calibration stream `[seed, 9000+step]`) |
| 300111-300120 | jobs_a5.json B formal seeds |
| 300201-300203 | jobs_a5.json R probe on task A (LoopNet init and training stream `[seed, 1000]`) |
| 300900 | train_r_probe.py dev sets (all lengths) |
| 1, 3, 7 | jobs_a5.json R probe positive control on P: the Stage A P384 R seeds that failed. Deliberately < 300000 because the control must replay the SAME init and training stream as the failed Stage A runs (`default_rng([seed, 1000])`, model seed = seed). |
