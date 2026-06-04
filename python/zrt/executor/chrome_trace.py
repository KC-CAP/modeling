"""Chrome Trace exporter — convert pipeline scheduling results to JSON.

Supports three input types:
  - ``PPStitchedTimeline``   : device x microbatch pipeline grid tasks
  - ``Timeline`` (per-device) : per-op trace from DAGScheduler
  - ``List[Timeline]``       : multi-device per-op trace, one per GPU

Chrome Trace event format (JSON array of dicts):

  {
    "ph":  "X",            // complete event (has duration)
    "name": "op_name",     // display name
    "cat":  "compute",     // category for filtering
    "pid":  device_id,     // process = GPU device
    "tid":  stage_or_stream,  // thread = virtual stage or compute/comm stream
    "ts":   123456.0,      // timestamp in microseconds
    "dur":  789.0,         // duration in microseconds
    "args": {"desc": "..."}
  }

  VPP (Virtual Pipeline Parallelism): devices with multiple virtual stages
  show each stage on a separate thread (tid).  This gives the "mbs blocks"
  visual grouping in Chrome Trace.

Usage
-----
>>> from python.zrt.executor.chrome_trace import ChromeTraceExporter
>>> exporter = ChromeTraceExporter()
>>> exporter.export_stitched(result, "pipeline.json")      # PP grid (device view)
>>> exporter.export_per_stage(timelines, "detail.json")    # per-op detail
>>> exporter.export_combined(result, timelines, "all.json") # combined
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.pp_stitcher import PPStitchedTimeline
    from python.zrt.executor.scheduler import ScheduledOp, Timeline


# ── colour palette ────────────────────────────────────────────────────────────

_FWD_COLOR = "#1a3a6b"
_BWD_COLOR = "#8FBC8F"
_BWD_DX_COLOR = "#6B8E6B"
_BWD_DW_COLOR = "#A8D8A8"

_COLORS: dict[str, str] = {
    "fwd_compute":        "good",
    "fwd_comm":           "olive",
    "fwd_p2p":            "terracotta",
    "bwd_compute":        "blue",
    "bwd_comm":           "purple",
    "bwd_p2p":            "magenta",
    "bwd_dx_compute":     "blue",
    "bwd_dw_compute":     "navy",
    "memory":             "yellow",
    "idle":               "grey",
    "bubble":             "light_grey",
    "warmup":             "orange",
    "cooldown":           "teal",
}

_NAMES: dict[str, str] = {
    "fwd_compute":        "[c]",
    "fwd_comm":           "[n]",
    "fwd_p2p":            "[p2p]",
    "bwd_compute":        "[c]",
    "bwd_comm":           "[n]",
    "bwd_p2p":            "[p2p]",
    "bwd_dx_compute":     "dX [c]",
    "bwd_dw_compute":     "dW [c]",
    "memory":             "MEM",
    "idle":               "idle",
    "bubble":             "bubble",
}


def _emit_template_replicas(
    events: list[dict],
    zero_dur_ops: list[dict],
    ops: list,
    base: float,
    window_dur: float,
    template_dur: float,
    stage: int,
    mb: int,
    phase: str,
    replicate: bool,
    mult: float,
    *,
    fwd_lat: float = 0.0,
    tid_offset: int = 0,
) -> None:
    """Emit *ops* as replicated template copies filling *window_dur*.

    The DAGScheduler *ops* represent one pass through the traced
    representative layers.  When the PP grid window is larger than
    *template_dur* (e.g. 1-2 traced layers in a 15-layer stage window),
    the template is replicated to fill the gap::

         window=[base … base+window_dur]
           [r=0: template] [r=1: template] [r=2: template] …

    Each replica gets ``L{replica_index}:`` prefixed to the op name so
    the trace viewer shows distinct per-layer blocks.

    Operators with zero or negative latency_us are skipped and collected
    in *zero_dur_ops* for later analysis.
    """
    if template_dur <= 0 or window_dur <= 0:
        return

    total_op_latency = sum(op.latency_us for op in ops)
    fill_ratio = total_op_latency / max(template_dur, 1.0)
    if (len(ops) < 5 and fill_ratio < 0.02) or fill_ratio < 0.005:
        _emit_copy(events, zero_dur_ops, ops, base, 0.0, stage, mb, phase,
                   replicate, mult, fwd_lat, 0,
                   tid_offset=tid_offset)
        return

    replicas = window_dur / template_dur
    full_copies = int(replicas)
    remainder = replicas - full_copies

    if len(ops) < 5 and full_copies > 50:
        _emit_copy(events, zero_dur_ops, ops, base, 0.0, stage, mb, phase,
                   replicate, mult, fwd_lat, 0,
                   tid_offset=tid_offset)
        return

    for r in range(full_copies):
        _emit_copy(events, zero_dur_ops, ops, base, r * template_dur, stage, mb, phase,
                   replicate, mult, fwd_lat, r, tid_offset=tid_offset)

    if remainder > 0.001:
        partial_offset = full_copies * template_dur
        partial_limit = partial_offset + remainder * template_dur
        _emit_copy(events, zero_dur_ops, ops, base, partial_offset, stage, mb, phase,
                   replicate, mult, fwd_lat, full_copies,
                   time_limit=partial_limit, tid_offset=tid_offset)


def _emit_copy(
    events: list[dict],
    zero_dur_ops: list[dict],
    ops: list,
    base: float,
    offset: float,
    stage: int,
    mb: int,
    phase: str,
    replicate: bool,
    mult: float,
    fwd_lat: float,
    layer_idx: int,
    *,
    time_limit: float | None = None,
    tid_offset: int = 0,
) -> None:
    for op in ops:
        if op.phase == "fwd" or not op.phase:
            rel_start = offset + op.start_us
        else:
            rel_start = offset + (op.start_us - fwd_lat)

        if rel_start < 0:
            continue
        if time_limit is not None and rel_start >= time_limit:
            continue

        if op.latency_us <= 0:
            zero_dur_ops.append({
                "op_type": op.op_type,
                "stream_type": op.stream_type,
                "phase": op.phase,
                "node_id": op.node_id,
                "mb": mb if replicate else -1,
                "layer": layer_idx,
            })
            continue

        dur_us = op.latency_us
        if time_limit is not None:
            end = rel_start + dur_us
            if end > time_limit:
                dur_us = max(0.0, time_limit - rel_start)
                if dur_us <= 0:
                    continue

        cat = "communication" if op.stream_type == "comm" else "compute"
        if replicate:
            name = f"m{mb}:L{layer_idx}:{op.phase}:{op.op_type}" if op.phase else f"m{mb}:L{layer_idx}:{op.op_type}"
        else:
            name = f"L{layer_idx}:{op.phase}:{op.op_type}" if op.phase else f"L{layer_idx}:{op.op_type}"

        events.append(ChromeTraceEvent(
            name=name,
            cat=cat,
            pid=stage,
            tid=tid_offset + (0 if op.stream_type != "comm" else 1),
            ts=(base + rel_start) * mult,
            dur=dur_us * mult,
            args={
                "phase": op.phase,
                "op_type": op.op_type,
                "stream_type": op.stream_type,
                "mb": mb,
                "layer": layer_idx,
            },
        ).to_dict())


@dataclass
class ChromeTraceEvent:
    """A single Chrome Trace complete event (ph="X")."""

    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict = field(default_factory=dict)
    color: str | None = None

    def to_dict(self) -> dict:
        d = {
            "ph": "X",
            "name": self.name,
            "cat": self.cat,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
            "dur": self.dur,
            "args": self.args,
        }
        if self.color:
            d["color"] = self.color
        return d


class ChromeTraceExporter:
    """Export pipeline scheduling results as Chrome Trace JSON.

    Parameters
    ----------
    time_unit : str
        "us" (default) or "ns".  Chrome Trace ``ts`` / ``dur`` fields
        are always in microseconds; ``ns`` multiplies by 1000.
    """

    def __init__(self, time_unit: str = "us") -> None:
        self._mult = 1000.0 if time_unit == "ns" else 1.0
        self._time_unit = time_unit
        self._zero_dur_ops: list[dict] = []

    # ── metadata helpers ──────────────────────────────────────────────────

    @staticmethod
    def _process_name_meta(pid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "ts": 0, "name": "process_name", "args": {"name": name}}

    @staticmethod
    def _thread_name_meta(pid: int, tid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "tid": tid, "ts": 0, "name": "thread_name", "args": {"name": name}}

    @staticmethod
    def _sort_index_meta(pid: int, sort_index: int) -> dict:
        return {"ph": "M", "pid": pid, "ts": 0, "name": "sort_index", "args": {"sort_index": sort_index}}

    @staticmethod
    def _thread_sort_index_meta(pid: int, tid: int, sort_index: int) -> dict:
        return {"ph": "M", "pid": pid, "tid": tid, "ts": 0, "name": "thread_sort_index", "args": {"sort_index": sort_index}}

    # ── device layout helpers (VPP support) ─────────────────────────────

    @staticmethod
    def _compute_device_layout(
        result: "PPStitchedTimeline",
    ) -> tuple[int, dict[int, list[int]], dict[tuple[int, int], int]]:
        """Compute VPP layout from stitched timeline.

        Returns
        -------
        vpp : int
            Max number of virtual stages per device.
        device_stages : dict[int, list[int]]
            device_id → sorted list of stage_ids on that device.
        stage_to_tid : dict[(device_id, stage_id), int]
            Maps (device, virtual_stage) → thread index on that device.
        """
        stages_per_dev: dict[int, set[int]] = {}
        for task in result.tasks:
            stages_per_dev.setdefault(task.stream_id, set()).add(task.stage_id)
        device_stages: dict[int, list[int]] = {
            d: sorted(s) for d, s in stages_per_dev.items()
        }
        vpp = max(len(s) for s in device_stages.values()) if device_stages else 1
        stage_to_tid: dict[tuple[int, int], int] = {}
        for d, stages in device_stages.items():
            for idx, s in enumerate(stages):
                stage_to_tid[(d, s)] = idx
        return vpp, device_stages, stage_to_tid

    def _grid_meta_events(
        self, pp: int, *, vpp: int = 1,
        device_stages: dict[int, list[int]] | None = None,
    ) -> list[dict]:
        meta: list[dict] = []
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            if vpp <= 1:
                meta.append(self._thread_name_meta(d, 0, "Grid Schedule"))
            else:
                stages = (device_stages or {}).get(d, [])
                for idx, s in enumerate(stages):
                    meta.append(self._thread_name_meta(d, idx, f"Stage {s}"))
                    meta.append(self._thread_sort_index_meta(d, idx, idx))
        return meta

    def _per_stage_meta_events(self, pp: int) -> list[dict]:
        meta: list[dict] = []
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            meta.append(self._thread_name_meta(d, 0, "Compute Ops"))
            meta.append(self._thread_sort_index_meta(d, 0, 0))
            meta.append(self._thread_name_meta(d, 1, "Comm Ops"))
            meta.append(self._thread_sort_index_meta(d, 1, 1))
        return meta

    def _combined_meta_events(
        self, pp: int, *, vpp: int = 1,
        device_stages: dict[int, list[int]] | None = None,
    ) -> list[dict]:
        meta: list[dict] = []
        detail_base = max(vpp, 2)
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            if vpp <= 1:
                meta.append(self._thread_name_meta(d, 0, "Grid Schedule"))
                meta.append(self._thread_sort_index_meta(d, 0, 0))
            else:
                stages = (device_stages or {}).get(d, [])
                for idx, s in enumerate(stages):
                    meta.append(self._thread_name_meta(d, idx, f"Stage {s}"))
                    meta.append(self._thread_sort_index_meta(d, idx, idx))
            meta.append(self._thread_name_meta(d, detail_base + 0, "Compute Ops"))
            meta.append(self._thread_sort_index_meta(d, detail_base + 0, vpp))
            meta.append(self._thread_name_meta(d, detail_base + 1, "Comm Ops"))
            meta.append(self._thread_sort_index_meta(d, detail_base + 1, vpp + 1))
        return meta

    # ── deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(events: list[dict]) -> list[dict]:
        """Merge X-events that share (pid, tid, ts, name, cat) into one with count."""
        meta = [e for e in events if e.get("ph") != "X"]
        x_events = [e for e in events if e.get("ph") == "X"]

        groups: dict[tuple, list[dict]] = {}
        for e in x_events:
            key = (e["pid"], e["tid"], e["ts"], e["name"], e.get("cat", ""))
            groups.setdefault(key, []).append(e)

        merged: list[dict] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                first = dict(group[0])
                first["args"] = dict(first.get("args", {}))
                first["args"]["count"] = len(group)
                merged.append(first)

        return meta + merged

    # ── public API ────────────────────────────────────────────────────────

    def export_stitched(
        self, result: "PPStitchedTimeline", path: str | None = None,
    ) -> str:
        """Export PPStitchedTimeline (device x microbatch grid).

        pid = device (stream_id), tid = stage chunk index within device.

        For VPP, each device gets one thread per virtual stage so the
        grid blocks for each stage appear in separate rows within the
        same GPU process.

        Metadata events name each process "GPU N" and threads
        "Stage N" (VPP) or "Grid Schedule".
        """
        events: list[dict] = []
        n_devices = result.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(result)
        events.extend(self._grid_meta_events(n_devices, vpp=vpp, device_stages=device_stages))

        for task in result.tasks:
            cat = self._grid_cat(task)
            name = self._name_for_task(task)
            color_val = self._color_for_task(task)
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=name,
                cat=cat,
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=task.latency_us * self._mult,
                color=color_val,
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "device": pid,
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        if result.warmup_us > 0:
            for d in range(n_devices):
                events.append(self._instant(
                    name=_NAMES.get("warmup", "warmup"),
                    pid=d, tid=0,
                    ts=result.warmup_us * self._mult,
                    args={"phase": "warmup", "dur_us": result.warmup_us},
                ))
        if result.cooldown_us > 0:
            cooldown_ts = (result.step_time_us - result.cooldown_us) * self._mult
            for d in range(n_devices):
                events.append(self._instant(
                    name=_NAMES.get("cooldown", "cooldown"),
                    pid=d, tid=0,
                    ts=cooldown_ts,
                    args={"phase": "cooldown", "dur_us": result.cooldown_us},
                ))

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_per_stage(
        self,
        timelines: list["Timeline"],
        path: str | None = None,
        *,
        M: int = 1,
        pp_stitched: "PPStitchedTimeline | None" = None,
        replicate: bool = True,
    ) -> str:
        """Export per-device DAGScheduler Timelines (per-op detail).

        ``timelines[d]`` maps to ``pid=d`` (one process per GPU).
        Within each device, ops on different streams are rendered on
        separate ``tid`` values (0 = "Compute Ops", 1 = "Comm Ops").

        When ``replicate=True`` and ``M > 1``, each microbatch's ops are
        expanded as time-offset replicas aligned with the pipeline grid
        schedule.  When ``replicate=False``, only one clean reference copy
        of each per-device timeline is exported.

        Zero-latency ops receive a minimum visible duration so they
        are not invisible in Chrome Trace.

        For VPP, each device's single Timeline already aggregates all
        virtual stages — the per-stage distinction is visible in the
        corresponding grid view (export_stitched / export_stitched_detailed).
        """
        pp = len(timelines)
        events: list[dict] = []
        events.extend(self._per_stage_meta_events(pp))

        grid_slot: dict[tuple[int, int, str], float] = {}
        grid_window: dict[tuple[int, int, str], float] = {}
        if pp_stitched is not None and M > 1:
            for task in pp_stitched.tasks:
                if task.phase in ("fwd", "bwd", "bwd_dx", "bwd_dw"):
                    grid_slot[(task.stage_id, task.mb_id, task.phase)] = task.start_us
                    grid_window[(task.stage_id, task.mb_id, task.phase)] = task.latency_us

        for s, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")
            if fwd_lat == 0.0 and bwd_lat == 0.0:
                fwd_lat = tl.total_latency_us
            stage_total = fwd_lat + bwd_lat

            fwd_ops = [op for op in tl.scheduled_ops if op.phase == "fwd" or not op.phase]
            bwd_ops = [op for op in tl.scheduled_ops if op.phase and op.phase != "fwd"]

            num_replicas = M if replicate else 1

            for m in range(num_replicas):
                fwd_base = grid_slot.get((s, m, "fwd"), m * stage_total)
                fwd_win  = grid_window.get((s, m, "fwd"), fwd_lat)
                bwd_base = grid_slot.get(
                    (s, m, "bwd"),
                    grid_slot.get((s, m, "bwd_dx"), m * stage_total + fwd_lat),
                )
                bwd_win  = grid_window.get(
                    (s, m, "bwd"),
                    grid_window.get((s, m, "bwd_dx"), bwd_lat),
                )

                # ── Fwd template replication ──
                _emit_template_replicas(
                    events, self._zero_dur_ops, fwd_ops, fwd_base, fwd_win, fwd_lat,
                    s, m, "fwd", replicate, self._mult,
                )

                # ── Bwd template replication ──
                _emit_template_replicas(
                    events, self._zero_dur_ops, bwd_ops, bwd_base, bwd_win, bwd_lat,
                    s, m, "bwd", replicate, self._mult,
                    fwd_lat=fwd_lat,
                )

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_combined(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Combined export: PP grid + per-device op detail shared on same pids.

        pid = device id (stream_id).

        Within each device:
          tid 0..vpp-1  = "Grid Schedule" per virtual stage (VPP) or
                          tid 0 = "Grid Schedule" (non-VPP)
          tid base+0    = "Compute Ops"    (per-op compute detail)
          tid base+1    = "Comm Ops"       (per-op communication detail)
          base = max(vpp, 2)

        Grid task categories use ``mb_{mb}`` so the same
        microbatch index gets the same colour across all stages.
        """
        n_devices = stitched.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(stitched)
        detail_base = max(vpp, 2)
        events: list[dict] = []
        events.extend(self._combined_meta_events(
            n_devices, vpp=vpp, device_stages=device_stages,
        ))

        for task in stitched.tasks:
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=task.latency_us * self._mult,
                color=self._color_for_task(task),
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "device": pid,
                    "view": "grid",
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        for d, tl in enumerate(timelines):
            for op in tl.scheduled_ops:
                if op.latency_us <= 0:
                    self._zero_dur_ops.append({
                        "op_type": op.op_type,
                        "stream_type": op.stream_type,
                        "phase": op.phase,
                        "node_id": op.node_id,
                        "mb": -1,
                        "layer": -1,
                    })
                    continue
                cat = "communication" if op.stream_type == "comm" else "compute"
                name = f"{op.phase}:{op.op_type}" if op.phase else op.op_type
                events.append(ChromeTraceEvent(
                    name=name,
                    cat=cat,
                    pid=d,
                    tid=detail_base + (0 if op.stream_type != "comm" else 1),
                    ts=op.start_us * self._mult,
                    dur=op.latency_us * self._mult,
                    args={
                        "phase": op.phase,
                        "op_type": op.op_type,
                        "stream_type": op.stream_type,
                        "view": "detail",
                    },
                ).to_dict())

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_stitched_detailed(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Stitched grid with per-device detail on separate tids (same pid).

        pid = device id (stream_id).
        Within each device:
          tid 0..vpp-1       = grid-level fwd/bwd blocks per virtual stage
          tid base + 0..N    = per-op detail from DAGScheduler Timeline
          base = max(vpp, 2)

        All microbatches share the same detail rows — they are distinguished
        by time offsets from the pipeline schedule, naturally serialised on
        their physical stream.
        """
        n_devices = stitched.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(stitched)
        detail_base = max(vpp, 2)
        events: list[dict] = []
        events.extend(self._combined_meta_events(
            n_devices, vpp=vpp, device_stages=device_stages,
        ))

        grid_index: dict[tuple[int, int, str], float] = {}
        grid_windows: dict[tuple[int, int, str], float] = {}
        for task in stitched.tasks:
            key = (task.stage_id, task.mb_id, task.phase)
            grid_index[key] = task.start_us
            grid_windows[key] = task.latency_us

        for task in stitched.tasks:
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=task.latency_us * self._mult,
                color=self._color_for_task(task),
                args={
                    "phase": task.phase, "mb": task.mb_id,
                    "stage": task.stage_id, "device": pid, "view": "grid",
                },
            ).to_dict())

        for d, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")

            fwd_ops = [op for op in tl.scheduled_ops if op.phase == "fwd" or not op.phase]
            bwd_ops = [op for op in tl.scheduled_ops if op.phase and op.phase != "fwd"]

            for m in range(stitched.M):
                fwd_base = grid_index.get((d, m, "fwd"), 0.0)
                fwd_win  = grid_windows.get((d, m, "fwd"), fwd_lat)
                bwd_base = grid_index.get(
                    (d, m, "bwd"),
                    grid_index.get((d, m, "bwd_dx"), 0.0),
                )
                bwd_win  = grid_windows.get(
                    (d, m, "bwd"),
                    grid_windows.get((d, m, "bwd_dx"), bwd_lat),
                )

                # ── Fwd template replication ──
                _emit_template_replicas(
                    events, self._zero_dur_ops, fwd_ops, fwd_base, fwd_win, fwd_lat,
                    d, m, "fwd", True, self._mult,
                    tid_offset=detail_base,
                )

                # ── Bwd template replication ──
                _emit_template_replicas(
                    events, self._zero_dur_ops, bwd_ops, bwd_base, bwd_win, bwd_lat,
                    d, m, "bwd", True, self._mult,
                    fwd_lat=fwd_lat, tid_offset=detail_base,
                )

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    # ── internals ─────────────────────────────────────────────────────────

    def _shift_overlap_comm_op(
        self,
        op,
        compute_index: dict[str, tuple[float, float]],
        base_us: float,
        original_ts: float,
    ) -> float:
        """Shift overlap comm op start time to visualize overlap in trace.

        Handles three overlap types:
        - CoC: K-wave shift (target_lat * (K-1) / K)
        - P2P/Ring-CP: full overlap (shift by target_lat)
        - Others: no shift

        Args:
            op: ScheduledOp with overlap_type, overlap_target, coc_tile_k
            compute_index: dict mapping node_id -> (start_us, latency_us) - NOT multiplied
            base_us: base timestamp for the current microbatch - NOT multiplied
            original_ts: original timestamp from scheduler - ALREADY multiplied by self._mult

        Returns:
            Shifted timestamp to show overlap in trace visualization.
        """
        overlap_type = op.overlap_type
        if overlap_type in ("none", ""):
            return original_ts

        target_key = op.overlap_target
        if not target_key or ":" not in target_key:
            return original_ts

        target_id = target_key.split(":", 1)[1]
        entry = compute_index.get(target_id)
        if entry is None:
            return original_ts

        target_start_us, target_lat = entry  # NOT multiplied
        if target_lat <= 0:
            return original_ts

        if overlap_type == "coc":
            coc_tile_k = getattr(op, "coc_tile_k", 4)
            if coc_tile_k <= 1:
                return original_ts
            shift_us = target_lat * (coc_tile_k - 1) / coc_tile_k  # NOT multiplied
            return original_ts - shift_us * self._mult

        elif overlap_type in ("p2p_overlap", "ring_cp"):
            # P2P should overlap with target compute
            # Strategy: shift P2P backward by target_latency to show overlap
            
            # Compute desired P2P start time (NOT multiplied)
            # P2P starts at same time as target (parallel execution)
            desired_start_us = base_us + target_start_us
            
            # Shift amount (NOT multiplied)
            # current = base_us + op.start_us
            # desired = base_us + target_start_us
            # shift = desired - current = target_start_us - op.start_us
            
            # But P2P might already start before target (scheduler put it earlier)
            # In that case, we want to shift forward to align
            
            # Alternative: shift backward by target_lat to show full overlap
            # This makes P2P appear during compute's execution window
            
            # Simple approach: shift by target_latency (shows P2P completely hidden)
            shift_us = target_lat
            
            # Apply shift (multiplied)
            shifted_ts = original_ts - shift_us * self._mult
            
# Clamp to >= 0 (can't have negative time in trace)
            return max(0.0, shifted_ts)

        return original_ts

    def _coc_shift_ts(
        self,
        op,
        compute_index: dict[str, tuple[float, float]],
        base_us: float,
        original_ts: float,
    ) -> float:
        """Legacy wrapper for backward compatibility."""
        return self._shift_overlap_comm_op(
            op, compute_index, base_us, original_ts
        )

    @staticmethod
    def _grid_cat(task) -> str:
        if task.phase == "fwd":
            return "fwd"
        return "bwd"

    @staticmethod
    def _name_for_task(task) -> str:
        mapping = {"fwd": "F", "bwd": "B", "bwd_dx": "B_dx", "bwd_dw": "B_dw"}
        prefix = mapping.get(task.phase, task.phase.upper()[:4])
        return f"{prefix} {task.mb_id}"

    @staticmethod
    def _color_for_task(task) -> str:
        if task.phase == "fwd":
            return _FWD_COLOR
        if task.phase == "bwd_dx":
            return _BWD_DX_COLOR
        if task.phase == "bwd_dw":
            return _BWD_DW_COLOR
        return _BWD_COLOR

    @staticmethod
    def _instant(name: str, pid: int, tid: int, ts: float, args: dict) -> dict:
        return {
            "ph": "i",
            "name": name,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "s": "g",   # scope = global
            "args": args,
        }

    def _build_doc(self, events: list[dict]) -> str:
        result: dict = {
            "traceEvents": events,
            "displayTimeUnit": "ns" if self._time_unit == "ns" else "ms",
        }
        if self._zero_dur_ops:
            from collections import Counter
            by_op_type = Counter(op["op_type"] for op in self._zero_dur_ops)
            by_phase = Counter(op["phase"] for op in self._zero_dur_ops)
            by_stream = Counter(op["stream_type"] for op in self._zero_dur_ops)
            result["zeroDurOps"] = {
                "total": len(self._zero_dur_ops),
                "byOpType": dict(by_op_type.most_common()),
                "byPhase": dict(by_phase),
                "byStreamType": dict(by_stream),
            }
            self._zero_dur_ops = []
        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def _write(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)