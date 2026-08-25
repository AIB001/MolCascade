"""A GPU that exists and goes unused must never be a silent outcome.

Two failures are possible when a screening run meets an accelerator, and they
call for opposite treatment.  A user who wrote ``--device auto`` is asking the
software to decide, so falling back to CPU is a correct decision -- but an
eight-hour CPU run that never says it could have been forty minutes is not.  A
user who wrote ``--device cuda`` has already decided, so quietly doing something
else is simply wrong, and the run has to stop.

No test here requires a GPU.  ``HostEnvironment`` is a plain measurement record,
so a machine with two cards is a literal, and the framework probe -- the one
question that genuinely needs a subprocess -- is stubbed at its own seam.
"""

from __future__ import annotations

import os

import pytest

from molcascade.environment.models import (
    AcceleratorVendor,
    CpuInfo,
    ExecutionPlan,
    GpuDevice,
    HostEnvironment,
    MemoryInfo,
    PlatformInfo,
)
from molcascade.errors import ConfigError
from molcascade.parallel.devices import (
    FrameworkProbe,
    describe_lanes,
    pinned_device,
    plan_lanes,
    probe_framework_gpu,
    reset_probe_cache,
    resolve_framework_lanes,
)
from molcascade.parallel.models import StageResources


def _environment(*, gpus: int = 0, cores: int = 8) -> HostEnvironment:
    devices = tuple(
        GpuDevice(
            index=index,
            vendor=AcceleratorVendor.NVIDIA,
            name=f"NVIDIA A100 #{index}",
            memory_total_mib=81_920,
        )
        for index in range(gpus)
    )
    return HostEnvironment(
        platform=PlatformInfo(
            system="Linux",
            python_version="3.12.0",
            python_implementation="CPython",
        ),
        cpu=CpuInfo(logical_cores=cores, usable_cores=cores),
        memory=MemoryInfo(total_bytes=64 * 1024**3),
        accelerator=AcceleratorVendor.NVIDIA if gpus else AcceleratorVendor.NONE,
        gpus=devices,
        plan=ExecutionPlan(
            device="cuda:0" if gpus else "cpu",
            gpu_count=gpus,
            recommended_workers=max(1, cores - 1),
            recommended_batch_size=4096,
            gpu_backends_runnable=bool(gpus),
            rationale="fixture",
        ),
    )


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    reset_probe_cache()


def test_two_cards_become_two_lanes() -> None:
    """One lane per card, not one per core.

    Two processes on one card contend for its memory and usually finish later
    than one would, so the useful parallelism when a GPU is in play is the
    number of cards.
    """

    plan = plan_lanes(_environment(gpus=2), device="auto")
    assert plan.devices == ("cuda:0", "cuda:1")
    assert plan.workers == 2
    assert plan.uses_gpu


def test_molecules_divide_evenly_across_the_detected_cards() -> None:
    """The user's actual request: spread the work over every GPU present.

    Shards are handed out round-robin over the lanes, so with four cards and a
    shard count that is a multiple of four each card gets exactly a quarter --
    and with any other count the difference is at most one shard.
    """

    lanes = plan_lanes(_environment(gpus=4), device="auto").devices
    assignment = [lanes[index % len(lanes)] for index in range(40)]
    per_card = {lane: assignment.count(lane) for lane in lanes}
    assert per_card == {"cuda:0": 10, "cuda:1": 10, "cuda:2": 10, "cuda:3": 10}


def test_no_gpu_falls_back_to_cpu_workers_and_says_so() -> None:
    plan = plan_lanes(_environment(gpus=0, cores=8), device="auto")
    assert set(plan.devices) == {"cpu"}
    assert plan.workers == 7  # one core left for the parent and the OS
    assert any("no NVIDIA device detected" in note for note in plan.notes)


def test_an_explicit_cpu_request_is_honoured_over_present_hardware() -> None:
    """Sharing a workstation is a legitimate reason to leave the cards alone."""

    plan = plan_lanes(_environment(gpus=2), device="cpu")
    assert not plan.uses_gpu
    assert any("--device cpu was requested" in note for note in plan.notes)


def test_specific_cards_can_be_named() -> None:
    plan = plan_lanes(_environment(gpus=4), device="cuda:0,2")
    assert plan.devices == ("cuda:0", "cuda:2")


def test_naming_a_card_the_machine_does_not_have_stops_the_run() -> None:
    with pytest.raises(ConfigError) as raised:
        plan_lanes(_environment(gpus=2), device="cuda:5")
    assert raised.value.code == "DEVICE_UNAVAILABLE"
    assert "[5]" in str(raised.value)


def test_requesting_cuda_on_a_machine_with_none_stops_the_run() -> None:
    """An explicit request is a statement about the hardware, not a preference."""

    with pytest.raises(ConfigError) as raised:
        plan_lanes(_environment(gpus=0), device="cuda")
    assert raised.value.code == "DEVICE_UNAVAILABLE"
    assert raised.value.hint is not None and "--device auto" in raised.value.hint


def test_a_malformed_device_request_is_rejected_before_anything_runs() -> None:
    with pytest.raises(ConfigError) as raised:
        plan_lanes(_environment(gpus=2), device="gpu0")
    assert raised.value.code == "DEVICE_REQUEST_INVALID"


def test_workers_can_lower_the_lane_count_but_not_raise_it() -> None:
    """There is nothing for a ninth process to be pinned to on an eight-card box."""

    limited = plan_lanes(_environment(gpus=4), device="auto", workers=2)
    assert limited.devices == ("cuda:0", "cuda:1")
    assert any("--workers 2" in note for note in limited.notes)

    asked_for_more = plan_lanes(_environment(gpus=2), device="auto", workers=16)
    assert asked_for_more.devices == ("cuda:0", "cuda:1")


def test_only_nvidia_devices_become_cuda_lanes() -> None:
    """The lanes are pinned with ``CUDA_VISIBLE_DEVICES``, which an AMD card ignores."""

    environment = _environment(gpus=0).model_copy(
        update={
            "accelerator": AcceleratorVendor.AMD,
            "gpus": (GpuDevice(index=0, vendor=AcceleratorVendor.AMD, name="Radeon"),),
        }
    )
    plan = plan_lanes(environment, device="auto")
    assert set(plan.devices) == {"cpu"}


def test_a_cpu_only_framework_degrades_and_records_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact case a ``torch+cpu`` wheel produces on a GPU node.

    Hardware detection is right -- the cards are there -- and the run still has
    to happen on CPU.  What must not happen is that it happens on CPU quietly.
    """

    monkeypatch.setattr(
        "molcascade.parallel.devices.probe_framework_gpu",
        lambda module, **_: FrameworkProbe(module, False, detail="torch is a +cpu build"),
    )
    resources = StageResources(workers=2, devices=("cuda:0", "cuda:1"), device_request="auto")
    resolved = resolve_framework_lanes(resources, module="torch", stage_id="t5_admet")

    assert not resolved.uses_gpu
    assert any("went unused" in note for note in resolved.notes)
    assert any("+cpu build" in note for note in resolved.notes)
    assert any("--device cuda" in note for note in resolved.notes)


def test_an_explicit_cuda_request_a_framework_cannot_honour_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "molcascade.parallel.devices.probe_framework_gpu",
        lambda module, **_: FrameworkProbe(module, False, detail="torch is a +cpu build"),
    )
    resources = StageResources(workers=2, devices=("cuda:0", "cuda:1"), device_request="cuda")

    with pytest.raises(ConfigError) as raised:
        resolve_framework_lanes(resources, module="torch", stage_id="t5_admet")
    error = raised.value
    assert error.code == "DEVICE_FRAMEWORK_UNAVAILABLE"
    assert "t5_admet" in str(error)
    assert error.hint is not None and "CUDA build of torch" in error.hint


def test_a_usable_framework_keeps_every_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "molcascade.parallel.devices.probe_framework_gpu",
        lambda module, **_: FrameworkProbe(module, True, 2, detail="CUDA 12.4"),
    )
    resources = StageResources(workers=2, devices=("cuda:0", "cuda:1"))
    resolved = resolve_framework_lanes(resources, module="torch")
    assert resolved.devices == ("cuda:0", "cuda:1")
    assert any("2 usable CUDA device(s)" in note for note in resolved.notes)


def test_a_cpu_plan_never_pays_for_a_framework_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing torch to confirm that CPU is CPU would cost seconds per stage."""

    def explode(*_: object, **__: object) -> FrameworkProbe:
        raise AssertionError("the framework must not be probed for a CPU plan")

    monkeypatch.setattr("molcascade.parallel.devices.probe_framework_gpu", explode)
    resources = StageResources(workers=4, devices=("cpu",) * 4)
    assert resolve_framework_lanes(resources, module="torch") is resources


def test_probing_an_unknown_framework_is_a_no_not_a_crash() -> None:
    probe = probe_framework_gpu("molcascade_no_such_framework")
    assert not probe.available
    assert probe.detail is not None


def test_the_probe_runs_out_of_process() -> None:
    """A CUDA init that segfaults must not take the screening run with it.

    Proven by asking a real subprocess for a module that cannot be imported:
    the answer comes back as data, and this interpreter is untouched.
    """

    import sys

    before = set(sys.modules)
    probe = probe_framework_gpu("torch")
    assert isinstance(probe.available, bool)
    assert "torch" not in set(sys.modules) - before


def test_pinning_shows_a_worker_exactly_one_card_and_then_restores() -> None:
    """``CUDA_VISIBLE_DEVICES`` is the only mechanism every backend obeys.

    ``admet_ai.ADMETModel`` accepts no device argument at all, so any card but
    the first would be unreachable without this.
    """

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    with pinned_device("cuda:3"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


def test_pinning_a_cpu_lane_touches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pinned_device("cpu"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_lanes_render_for_a_human() -> None:
    assert describe_lanes(("cuda:0", "cuda:1")) == "2 GPU lane(s): cuda:0, cuda:1"
    assert describe_lanes(("cpu", "cpu", "cpu")) == "3 CPU worker(s)"
