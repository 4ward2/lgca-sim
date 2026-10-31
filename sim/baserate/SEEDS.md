# Seeds used under sim/baserate (all >= 800000; 12345 and 24680 never used)

| seed | where |
|---|---|
| 800001 | test_baserate.py sampler shape/density tests |
| 800002 | test_baserate.py corrupted-table test |
| 800003 | test_baserate.py small end-to-end run |
| 800004 | local throughput measurement, uniform |
| 800005 | local throughput measurement, density |
| 800006 | zero-propagation control (--zeroprop seed) |
| 810000 + 1000*k + j | GitHub shards, k = 0..5 in order P-uniform, P-density, A-uniform, A-density, M-uniform, M-density; j = shard index (see jobs.json) |
