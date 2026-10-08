"""The request loop shared by the two vLLM backends: admit a few requests per engine step up to a cap, step the
engine, hand finished outputs back with their wall-clock time."""
import time


def run_requests(eng, jobs, prompt_of, sp_of, on_done, cap, admit_per_step=8, quiet=False, progress_every=60, tag="doubtless"):
    """jobs: list of (index, rep).  prompt_of(i) / sp_of(i, k) build the request; on_done(i, k, token_ids, wall_s)
    consumes a finished one.  Requests are added at most `admit_per_step` per step and at most `cap` in flight."""
    jobs = list(jobs)
    live, t0, last, gen, done = {}, time.time(), time.time(), 0, 0
    total = len(jobs)

    def admit():
        n_new = 0
        while jobs and len(live) < cap and n_new < admit_per_step:
            i, k = jobs.pop(0); n_new += 1
            rid = "%s|%d|%d|%d" % (tag, i, k, int(time.time() * 1000) % 100000)
            eng.add_request(rid, prompt_of(i), sp_of(i, k))
            live[rid] = (i, k, time.time())

    admit()
    while live:
        for o in eng.step():
            if not (o.finished and o.request_id in live):
                continue
            i, k, ts = live.pop(o.request_id)
            out = list(o.outputs[0].token_ids)
            gen += len(out); done += 1
            on_done(i, k, out, time.time() - ts)
        admit()
        if progress_every and not quiet and time.time() - last > progress_every:
            print(">> %s: %d done / %d, %d in flight, %.1f tok/s (finished runs)"
                  % (time.strftime("%H:%M:%S"), done, total, len(live), gen / (time.time() - t0)), flush=True)
            last = time.time()
