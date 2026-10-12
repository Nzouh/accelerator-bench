"""Sample NVML power in a separate phase from latency measurement."""
from __future__ import annotations

import threading
import time

import numpy as np


def integrate_power(samples, start, end):
    """Trapezoidal integral with interpolated endpoint samples. Seconds × W = J."""
    if end <= start:
        raise ValueError("Power interval must be positive")
    if len(samples) < 2 or samples[0]["t_s"] > start or samples[-1]["t_s"] < end:
        raise ValueError("Telemetry must bracket the measurement interval")
    times = np.array([s["t_s"] for s in samples])
    watts = np.array([s["power_w"] for s in samples])
    if not np.isfinite(watts).all() or (watts < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("Invalid power samples")
    interior = (times > start) & (times < end)
    t = np.concatenate(([start], times[interior], [end]))
    p = np.concatenate(([np.interp(start, times, watts)], watts[interior], [np.interp(end, times, watts)]))
    return float(np.sum(np.diff(t) * (p[:-1] + p[1:]) / 2))


class PowerSampler:
    def __init__(self, index=0, interval=0.05):
        self.interval = interval
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.index = index

    def __enter__(self):
        try:
            import pynvml as nv
            self.nv = nv
            nv.nvmlInit()
            self.handle = nv.nvmlDeviceGetHandleByIndex(self.index)
            self.sample()
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return self

    def sample(self):
        power = self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000
        sample = {"t_s": time.perf_counter(), "power_w": power}
        for key, getter in [
            ("temperature_c", lambda: self.nv.nvmlDeviceGetTemperature(self.handle, 0)),
            ("graphics_clock_mhz", lambda: self.nv.nvmlDeviceGetClockInfo(self.handle, 0)),
            ("gpu_utilization_percent", lambda: self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu),
        ]:
            try:
                sample[key] = getter()
            except Exception:
                sample[key] = None
        self.samples.append(sample)

    def loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                break

    def __exit__(self, *args):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join()
            try:
                self.sample()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
        if hasattr(self, "nv"):
            try:
                self.nv.nvmlShutdown()
            except Exception:
                pass


def energy_measurement(fn, seconds=5.0, device_index=0):
    import torch
    if seconds <= 0:
        return {"status": "disabled"}
    # Bring this exact workload to a sustained state before telemetry begins.
    deadline = time.perf_counter() + 1.0
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            for _ in range(100):
                fn()
            torch.cuda.synchronize()
        count = 0
        with PowerSampler(device_index) as sampler:
            start = time.perf_counter()
            while time.perf_counter() - start < seconds:
                for _ in range(100):
                    fn()
                torch.cuda.synchronize()
                count += 100
            end = time.perf_counter()
    if sampler.error:
        return {"status": "unavailable", "reason": sampler.error}
    joules = integrate_power(sampler.samples, start, end)
    for sample in sampler.samples:
        sample["t_s"] -= start
    return {
        "status": "measured", "scope": "NVML-reported GPU device power; not whole laptop",
        "duration_s": end - start, "executions": count, "gpu_energy_j": joules,
        "mean_gpu_power_w": joules / (end - start),
        "gpu_energy_mj_per_execution": 1000 * joules / count,
        "wall_ms_per_execution": 1000 * (end - start) / count,
        "sample_interval_s_requested": sampler.interval,
        "samples": sampler.samples,
        "caveat": "Gross energy including baseline/device activity, sampled sensor estimate; no idle subtraction. Repeated readings can share the sensor's internal averaging window.",
    }
