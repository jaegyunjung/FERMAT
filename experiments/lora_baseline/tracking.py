"""Allowlisted aggregate metrics; local JSONL remains authoritative if W&B fails."""
import atexit
import json
import math
import os
import time
from pathlib import Path

PREFIXES = ('progress/', 'time/', 'train/', 'validation/', 'test/', 'gate/', 'gpu/', 'audit/')


class Tracker:
    def __init__(self, out, mode='disabled', project=None, entity=None, name=None, config=None):
        self.out, self.mode = Path(out), mode
        self.run, self.closed = None, False
        self.started = time.monotonic()
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = dict(mode=mode, connected=False)
        if mode != 'disabled':
            try:
                os.environ['WANDB_ERROR_REPORTING'] = 'false'
                import wandb
                settings = wandb.Settings(console='off', disable_git=True, disable_code=True,
                    x_disable_meta=True, x_disable_stats=True, x_save_requirements=False,
                    init_timeout=20, ignore_globs=['*.log', '*.patch', '*.py', '*.pt', '*.csv', '*.npy', '*.npz'])
                self.run = wandb.init(project=project, entity=entity, name=name, mode=mode,
                    dir=str(self.out), save_code=False, settings=settings, config=config or {})
                self.status.update(connected=True, run_id=self.run.id, url=self.run.url)
            except Exception as exc:
                self.status.update(error_type=type(exc).__name__, local_only=True)
        self._status()
        atexit.register(self.finish, False)

    def _status(self):
        (self.out / 'tracking.json').write_text(json.dumps(self.status, indent=2) + '\n')

    def log(self, metrics):
        if any(not k.startswith(PREFIXES) or not isinstance(v, (float, int, bool))
               or not math.isfinite(v) for k, v in metrics.items()):
            raise ValueError('Tracking accepts finite numeric aggregates with approved prefixes only')
        metrics = dict(metrics, **{'time/elapsed_seconds': time.monotonic() - self.started})
        with (self.out / 'aggregate_history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        if self.run is not None:
            try:
                self.run.log(metrics)
            except Exception as exc:
                self.status.update(error_type=type(exc).__name__, delivery_failed=True)
                self._status()

    def finish(self, success=True):
        if self.closed:
            return
        self.closed = True
        self.status.update(finished=True, success=success)
        if self.run is not None:
            try:
                self.run.finish(exit_code=0 if success else 1)
            except Exception as exc:
                self.status.update(finish_error=type(exc).__name__)
        self._status()


def add_arguments(parser):
    parser.add_argument('--tracking', choices=('disabled', 'offline', 'online'), default='disabled')
    parser.add_argument('--wandb-project', default='fermat-lora-baseline')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--run-name', default=None)
    return parser
