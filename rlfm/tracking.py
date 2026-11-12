"""Thin wrapper around the nb experiment tracker; falls back to local JSONL logging when nb is absent."""
import json
import os
import time

try:
    import nb  # type: ignore

    HAVE_NB = True
except Exception:  # nb not installed (e.g. local tests): a stub so that @nb.core still works
    import types

    nb = types.SimpleNamespace(core=lambda fn: fn)
    HAVE_NB = False


class LocalRun:
    """Minimal stand-in for an nb run: writes metrics to out_dir/metrics.jsonl."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "metrics.jsonl")

    def log(self, step: int, **metrics):
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")

    def figure(self, *args, **kwargs):
        pass

    def finish(self, **metrics):
        if metrics:
            self.log(step=-1, **metrics)


class Run:
    """Logs to nb (when used) and always to the local JSONL file."""

    def __init__(self, out_dir: str, use_nb: bool, **start_kwargs):
        self.local = LocalRun(out_dir)
        self.remote = nb.start(**start_kwargs) if (use_nb and HAVE_NB) else None

    def log(self, step: int, **metrics):
        self.local.log(step, **metrics)
        if self.remote is not None:
            self.remote.log(step=step, **metrics)
        print(f"[{step}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)

    def finish(self, **metrics):
        self.local.finish(**metrics)
        if self.remote is not None:
            self.remote.finish(**metrics)
