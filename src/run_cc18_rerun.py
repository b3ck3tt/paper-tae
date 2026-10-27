"""Driver for the CC18 API re-run of the P4 configurations (27 Oct 2026, protocol decision). Resumable: run it again
after any interruption; finished tasks are skipped (the harness keeps per-task results), tasks that failed on
network / server errors are retried in the next pass, tasks the model answered empty 3 times stay "final" missing.

Jobs (each = one harness process, all in parallel):
  A  results/results_recommender_paper4_cc18all  72 tasks, all toolkits    9 configs (no Qwen, no DeepSeek: 32k limit)
  B  results/results_recommender_paper4_cc18sk   69 tasks, sklearn          11 configs (primary package)
  T  results/results_recommender_paper4_turbo50  50 long-tail tasks         2 Llama-Turbo configs (control)
Spend guard: every harness call checks the summed api_log.jsonl of all three dirs and stops at $200.
progress.md (repo root) is rewritten every ~2 h and at the end.
Usage: python recommender/run_cc18_rerun.py"""
import os, sys, time, json, subprocess, joblib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DIRS = {"A": "results/results_recommender_paper4_cc18all", "B": "results/results_recommender_paper4_cc18sk",
        "T": "results/results_recommender_paper4_turbo50"}
CLAUDE = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-sonnet-5-think", "claude-opus-4-8", "claude-fable-5"]
OPENAI = ["gpt-5", "gpt-5-mini"]
L70, L8 = "deepinfra:meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepinfra:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEEP = ["deepinfra:deepseek-ai/DeepSeek-V3", L70, L8]
QWEN = "deepinfra:Qwen/Qwen2.5-72B-Instruct"
# DeepSeek-V3 and Qwen2.5-72B are served with a 32,768-token limit on DeepInfra (DeepSeek: HTTP 400 "max_num_tokens
# (32768)" despite the listed 163,840) -> both n/a on package A (prompt ~48k tokens), run on package B only.
# Fable on package A: every attempt ends at max_tokens (13.4k, all spent on its forced adaptive thinking) with an
# empty ranking (~$0.7 per attempt) -> stopped after 8 attempts on 27 Oct 2026, n/a (not run on this pack).
JOBS = ([("A", m) for m in CLAUDE + OPENAI + [L70, L8] if m != "claude-fable-5"] + [("B", m) for m in CLAUDE + OPENAI + DEEP + [QWEN]]
        + [("T", L70), ("T", L8)])
ENV = {**os.environ, "P4_BUDGET_DIRS": ":".join(DIRS.values()), "P4_BUDGET_STOP": "200", "P4_PRICES_2026_10": "1",
       "PYTHONUNBUFFERED": "1"}


def safe(m):
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", m)


def status(pkg, model):
    d = Path(DIRS[pkg]); n = len(joblib.load(d / "pack.joblib")["packs"])
    f = d / f"llm_{safe(model)}.joblib"
    res = joblib.load(f) if f.exists() else {}
    ok = sum(1 for v in res.values() if v.get("ranking")); trunc = sum(1 for v in res.values() if v.get("truncated"))
    final = sum(1 for v in res.values() if v.get("final")); err = sum(1 for v in res.values() if "error" in v and not v.get("final"))
    return n, ok, trunc, final, err


def write_progress(note=""):
    from llm_paper4 import spent_so_far
    os.environ.update({k: ENV[k] for k in ("P4_BUDGET_DIRS", "P4_PRICES_2026_10")})
    lines = [f"# CC18 re-run progress — {time.strftime('%Y-%m-%d %H:%M')} {note}", "",
             f"Spend so far (list prices, from api_log.jsonl): **${spent_so_far():.2f}** (stop at $200, cap $220)", "",
             "| pkg | model | tasks | ranked | of which truncated | final missing | pending errors |", "|---|---|---|---|---|---|---|"]
    for pkg, m in JOBS:
        n, ok, tr, fin, err = status(pkg, m)
        lab = m.replace("deepinfra:", "").replace("-Instruct-Turbo", "-Turbo (FP8)")
        lines.append(f"| {pkg} | {lab} | {n} | {ok} | {tr} | {fin} | {err} |")
    Path("progress.md").write_text("\n".join(lines) + "\n")


def running_jobs():
    """(P4_OUT, model) of harness processes already running (read from their environment)."""
    out = set()
    for pid in subprocess.run(["pgrep", "-f", "llm_paper4.py run"], capture_output=True, text=True).stdout.split():
        cmd = subprocess.run(["ps", "eww", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        import re
        o = re.search(r"P4_OUT=(\S+)", cmd); m = re.search(r"llm_paper4\.py run (\S+)", cmd)
        if o and m:
            out.add((o.group(1), m.group(1)))
    return out


def apply_reasks():
    """protocol decision (27 Oct): re-ask the tasks listed in results/cc18_reask.json whose earlier longer partial ranking was
    lost (pre-f17e4a5 harness). For jobs not currently running, the old entry is archived to
    <dir>/reask_archive.jsonl and removed, so the next harness run asks again (up to 3 new attempts)."""
    f = Path("results/cc18_reask.json")
    if not f.exists():
        return
    todo = json.loads(f.read_text()); running = running_jobs(); left = []
    for k, m, t in todo:
        if (DIRS[k], m) in running:
            left.append([k, m, t]); continue
        rf = Path(DIRS[k]) / f"llm_{safe(m)}.joblib"
        res = joblib.load(rf) if rf.exists() else {}
        old = res.pop(t, None) or res.pop(str(t), None)
        if old is not None:
            with open(Path(DIRS[k]) / "reask_archive.jsonl", "a") as fh:
                fh.write(json.dumps({"tid": t, "model": m, "archived": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     "old": old}, default=str) + "\n")
            joblib.dump(res, rf)
    f.write_text(json.dumps(left))


def needs_work(k, m):
    n, ok, tr, fin, err = status(k, m)
    return ok + fin < n


def main():
    last, idle, p = 0, 0, 0
    while True:
        apply_reasks()
        todo = [(k, m) for k, m in JOBS if needs_work(k, m)]
        if not todo:
            break
        before = sum(status(k, m)[1] + status(k, m)[3] for k, m in JOBS)
        p += 1
        print(f"=== pass {p}: {len(todo)} jobs ({time.strftime('%H:%M')}) ===", flush=True)
        procs = []
        running = running_jobs()
        for k, m in todo:
            if (DIRS[k], m) in running:              # already running (e.g. survived a driver restart): don't duplicate
                continue
            log = open(Path(DIRS[k]) / f"run_{safe(m)}.log", "a")
            procs.append(subprocess.Popen([sys.executable, "recommender/llm_paper4.py", "run", m],
                                          env={**ENV, "P4_OUT": DIRS[k]}, stdout=log, stderr=subprocess.STDOUT))
        while any(pr.poll() is None for pr in procs) or running_jobs():
            pending = json.loads(Path("results/cc18_reask.json").read_text()) if Path("results/cc18_reask.json").exists() else []
            if pending and len(running_jobs()) < len(JOBS):
                apply_reasks()                       # re-ask entries become workable as soon as their job is idle
                for k, m in JOBS:
                    if needs_work(k, m) and (DIRS[k], m) not in running_jobs():
                        log = open(Path(DIRS[k]) / f"run_{safe(m)}.log", "a")
                        procs.append(subprocess.Popen([sys.executable, "recommender/llm_paper4.py", "run", m],
                                                      env={**ENV, "P4_OUT": DIRS[k]}, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(60)
            if time.time() - last > 7200:
                write_progress(f"(pass {p})"); last = time.time()
        if any(pr.returncode == 3 for pr in procs):
            write_progress("— BUDGET STOP at $200"); print("BUDGET STOP", flush=True); return
        write_progress(f"(after pass {p})"); last = time.time()
        after = sum(status(k, m)[1] + status(k, m)[3] for k, m in JOBS)
        idle = 0 if after > before else idle + 1     # no progress = network down / provider outage
        if idle >= 36:                               # ~18 h without progress -> give up, resumable later
            write_progress("— STOPPED: no progress for ~18 h (network?) — rerun the driver to resume"); return
        time.sleep(300 if idle == 0 else 1800)
    write_progress("— DONE"); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
