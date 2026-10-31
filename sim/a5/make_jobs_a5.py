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
report job: aggregates everything (needs all).
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


def build():
    jobs = []
    for task, ks in B_CONFIGS:
        for K in ks:
            pilot = f"b-{task}-k{K}-pilot"
            for name, seeds, needs, formal in ((pilot, PILOT_SEEDS, [], False),
                                               (f"b-{task}-k{K}-formal", FORMAL_SEEDS, [pilot], True)):
                cmds = [f"{ENV} python -m sim.a5.train_b --task {task} --K {K} --seed {s} --threads 4 "
                        f"--out out/{name}/s{s}" for s in seeds]
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
    return dict(schema_version=1, status="exploratory-A5; not locked (brief_A5 pending Fable audit)",
                artifact_contract="Each dependency is downloaded into in/<consumer>/<dependency>/. All writes remain out/<job>/. "
                                  "R segments resume from in/<job>/<previous segment>/ckpt.pt.",
                jobs=jobs)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_a5.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
    print("wrote", path, len(build()["jobs"]), "jobs")
