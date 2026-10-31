"""Generate sim/a5/jobs_a5.json (A4 DAG schema {"jobs":[{name, needs, command}]}).  Run: python -m sim.a5.make_jobs_a5

Dependencies of job j are downloaded by tools/gh_dag.py into in/<j>/<dep>/; every job writes only to out/<job>/.

B jobs (train_b): one job per (task, K, set).  Set `pilot` = 3 seeds (300101..300103); set `formal` = 10 seeds
(300111..300120), needs the pilot job of the same (task, K).  Seeds of one job run sequentially into out/<job>/s<seed>/.
  m1d          K in {2, 4, 16}   (K=2/4 sanity scale per Fable theorem 2; K=16 for completeness)
  m2d_dense    K in {2, 4, 16}   (K=4 = representability threshold, theorem 3; K=2 conjectured impossible; K=16 control)
  m2d_last     K in {2, 4, 16}
  delay3       K in {1, 5, 8}    (d-2 not representable, d+2 representable, 2^d one-hot)
  delay4       K in {2, 6, 16}
R probe jobs (train_r_probe): A (task A, cos head) x decay exponent {1, 3, 10, 30} x seeds 300201..300203, and the positive
  control P (seeds 1, 3, 7 = the three Stage A P384 R seeds that failed: n=16 exact 0.039 / 0.001 / 0.025, runs/final/
  P_R_G384_alpha0.5_beta2.0_s{1,3,7}/test.json; head=const so the first 4000 steps replay Stage A's constant-lr run).
  Each 60000-step run is split into SEGMENTS segments of 60000/SEGMENTS steps chained by needs (ckpt.pt in the artifact).
SAT jobs (k2_sat, 4 h limit each, `pip install -q python-sat` first; each writes out/<job>/k.cnf.result.json):
  m2d n in {2,3} K=2 targets {last, all}; m2d n=2 K=4 targets last (must be SAT); delay d=2 K in {2,3,4} n<=5;
  delay d=3 K in {2,3,4,5} n<=7.                                                        (12 jobs)
report job: aggregates everything (needs all B jobs and final R segments).
verdict job (DESIGN CHOICE: a separate job, not part of report): `python -m sim.a5.verdict_a5 in/verdict --out out/verdict`
  applies the pre-registered rules P1-P4, C, R1, R2; needs report + every SAT job + every B formal job + every final R
  segment (pilots are not needed: formal needs pilot, so they are transitively complete and they carry no test.json).
Pilot B jobs run with --no_test (never touch test seed 300778).
"""
import json
import os

SEGMENTS = 4
STEPS = 60000
DECAYS = (0, 1, 3, 10, 30)  # 0 = wd 0 (pure longer training); Claude 2026-10-31
A_SEEDS = (300201, 300202, 300203)
P_SEEDS = (1, 3, 7)
PILOT_SEEDS = (300101, 300102, 300103)
FORMAL_SEEDS = tuple(range(300111, 300121))
B_CONFIGS = [("m1d", (2, 4, 16)), ("m2d_dense", (2, 4, 16)), ("m2d_last", (2, 4, 16)),
             ("delay3", (1, 5, 8)), ("delay4", (2, 6, 16))]
ENV = "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
SAT_TIME_LIMIT = 14400
# (job name, k2_sat arguments)
SAT_JOBS = ([(f"sat-m2d-n{n}-k2-{tg}", f"--task m2d --n {n} --K 2 --targets {tg}") for n in (2, 3) for tg in ("last", "all")]
            + [("sat-m2d-n2-k4-last", "--task m2d --n 2 --K 4 --targets last")]
            + [(f"sat-delay-d2-k{K}", f"--task delay --d 2 --n 5 --K {K}") for K in (2, 3, 4)]
            + [(f"sat-delay-d3-k{K}", f"--task delay --d 3 --n 7 --K {K}") for K in (2, 3, 4, 5)])


def build():
    jobs = []
    for task, ks in B_CONFIGS:
        for K in ks:
            pilot = f"b-{task}-k{K}-pilot"
            for name, seeds, needs, formal in ((pilot, PILOT_SEEDS, [], False),
                                               (f"b-{task}-k{K}-formal", FORMAL_SEEDS, [pilot], True)):
                cmds = [f"{ENV} python -m sim.a5.train_b --task {task} --K {K} --seed {s} --threads 4 "
                        f"--out out/{name}/s{s}" + ("" if formal else " --no_test") for s in seeds]
                jobs.append(dict(name=name, needs=needs, command=" && ".join(cmds), kind="B", formal=formal,
                                 task=task, K=K, seeds=list(seeds)))
    runs = []
    for d in DECAYS:
        for s in A_SEEDS:
            runs.append(("A", s, d, "const"))  # Stage A literally ran sched const (Claude 2026-10-31)
    for d in DECAYS:
        for s in P_SEEDS:
            runs.append(("P", s, d, "const"))
    seg = STEPS // SEGMENTS
    last_names = []
    for task, s, d, head in runs:
        base = f"r{task}-d{d}-s{s}"
        prev = None
        for g in range(1, SEGMENTS + 1):
            name = f"{base}-g{g}"
            cmd = (f"{ENV} python -m sim.a5.train_r_probe --task {task} --seed {s} --decay {d} --head {head} "
                   f"--steps {STEPS} --stop_at {g * seg} --threads 4 --out out/{name}")
            if prev:
                cmd += f" --resume in/{name}/{prev}"
            jobs.append(dict(name=name, needs=[prev] if prev else [], command=cmd, kind="R", task=task, seed=s,
                             decay=d, head=head, segment=g, segments=SEGMENTS, control=(task == "P")))
            prev = name
        last_names.append(prev)
    b_names = [j["name"] for j in jobs if j["kind"] == "B"]
    jobs.append(dict(name="report", needs=b_names + last_names, kind="report",
                     command="python -m sim.a5.report_a5 --input in/report --out out/report"))
    sat_names = []
    for name, args in SAT_JOBS:
        jobs.append(dict(name=name, needs=[], kind="SAT",
                         command=f"pip install -q python-sat && mkdir -p out/{name} && python -m sim.a5.k2_sat {args} "
                                 f"--time-limit {SAT_TIME_LIMIT} --out out/{name}/k.cnf"))
        sat_names.append(name)
    formal_names = [j["name"] for j in jobs if j["kind"] == "B" and j["formal"]]
    jobs.append(dict(name="verdict", needs=["report"] + sat_names + formal_names + last_names, kind="verdict",
                     command="python -m sim.a5.verdict_a5 in/verdict --out out/verdict"))
    return dict(schema_version=1, status="exploratory-A5; not locked (brief_A5 pending Fable audit)",
                artifact_contract="Each dependency is downloaded into in/<consumer>/<dependency>/. All writes remain out/<job>/. "
                                  "R segments resume from in/<job>/<previous segment>/ckpt.pt.",
                jobs=jobs)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_a5.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
    print("wrote", path, len(build()["jobs"]), "jobs")
