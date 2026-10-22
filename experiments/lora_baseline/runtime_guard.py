"""Optional WSL/Windows guard. Checks new incidents; never changes OS/GPU settings.

Uses the existing workstation's conservative 400 W / 1500 MHz operating profile.
Historical incidents remain in the preflight report; fresh System events abort.
"""
import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def validate_sample(values):
    limit, draw, temp, clock, memory = map(float, values)
    if not all(math.isfinite(x) and x >= 0 for x in (limit, draw, temp, clock, memory)):
        raise RuntimeError('Invalid GPU telemetry')
    if not (abs(limit - 400) < .5 and draw < 330 and temp < 70 and clock <= 1530 and memory < 30000):
        raise RuntimeError('GPU operating limit exceeded')
    return dict(zip(('power_limit_w', 'power_w', 'temperature_c', 'clock_mhz', 'memory_mib'),
                    (limit, draw, temp, clock, memory)))


class RuntimeGuard:
    def __init__(self, profile, out, tracker=None):
        self.enabled = profile is not None
        self.tracker, self.out = tracker, Path(out)
        self.last_sample, self.last_host, self.boot = 0., 0., None
        self.since = datetime.now(timezone.utc).isoformat()
        if self.enabled:
            self.profile = json.loads(Path(profile).read_text(encoding='utf-8-sig'))
            if self.profile['exit_code'] != 0 or self.profile['max_clock_mhz'] != 1500:
                raise RuntimeError('Validated GPU profile required')
            self.script = subprocess.run(['wslpath', '-w', str(Path(__file__).with_name('host_health.ps1'))],
                                         capture_output=True, text=True, check=True).stdout.strip()
            self.check(force=True)

    def check(self, force=False):
        if not self.enabled or (not force and time.monotonic() - self.last_sample < 2):
            return
        raw = subprocess.run(['/mnt/c/Windows/System32/nvidia-smi.exe',
            '--query-gpu=power.limit,power.draw,temperature.gpu,clocks.current.graphics,memory.used,driver_version',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=6, check=True).stdout
        rows = list(csv.reader(raw.strip().splitlines()))
        if len(rows) != 1 or rows[0][5].strip() != self.profile['driver_version']:
            raise RuntimeError('GPU count or driver mismatch')
        metrics = validate_sample(rows[0][:5])
        if force or time.monotonic() - self.last_host >= 10:
            raw = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
                '-NoProfile', '-NonInteractive', '-File', self.script, '-SinceIso', self.since],
                capture_output=True, text=True, timeout=12, check=True).stdout
            health = json.loads(raw.lstrip('\ufeff'))
            if health['reboot_required'] or health['events']:
                raise RuntimeError('New host incident or pending reboot: ' + json.dumps(health))
            if self.boot is not None and self.boot != health['boot']:
                raise RuntimeError('Host rebooted')
            self.boot = health['boot']
            if datetime.fromisoformat(self.profile['updated_at']) <= datetime.fromisoformat(self.boot.replace('Z', '+00:00')):
                raise RuntimeError('GPU profile predates current boot')
            self.last_host = time.monotonic()
        self.last_sample = time.monotonic()
        with (self.out / 'gpu_telemetry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix=time.time(), **metrics)) + '\n')
        if self.tracker is not None:
            self.tracker.log({'gpu/' + k: v for k, v in metrics.items()})
