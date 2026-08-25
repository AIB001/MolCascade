"""Hardware detection.

The one behaviour that matters more than any individual field is that this
module never raises.  It runs before a screening run and while generating the
browser payload, so a host that answers nothing must still produce a complete
record rather than an exception.  Most of the tests below therefore feed
detection a hostile answer -- a truncated ``/proc`` file, an ``nvidia-smi`` that
prints an error page, a cgroup that says ``max`` -- and assert that a valid
:class:`HostEnvironment` still comes out the other side.

The parsers are exercised directly with captured real output.  ``nvidia-smi``
CSV in particular is transcribed from a driver's actual response, including the
``[N/A]`` that appears where a field is unsupported and is emphatically not an
integer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.environment import detect as detect_module
from molcascade.environment import detect_environment
from molcascade.environment.models import (
    AcceleratorVendor,
    CpuInfo,
    GpuDevice,
    HostEnvironment,
    MemoryInfo,
)

_GIB = 1024 * 1024 * 1024


# ------------------------------------------------------------ whole-host probe


def test_detecting_this_machine_produces_a_complete_record() -> None:
    environment = detect_environment()

    assert isinstance(environment, HostEnvironment)
    assert environment.schema_version == 1
    assert environment.platform.python_version
    assert environment.plan.recommended_workers >= 1
    assert environment.plan.recommended_batch_size >= 1


def test_the_record_round_trips_through_json() -> None:
    """It goes into run provenance and into the browser, so it must serialise."""

    environment = detect_environment()

    assert HostEnvironment.model_validate_json(environment.model_dump_json()) == environment


def test_skipping_the_gpu_probe_reports_no_accelerator_and_says_so() -> None:
    environment = detect_environment(include_gpus=False)

    assert environment.gpus == ()
    assert environment.accelerator is AcceleratorVendor.NONE
    assert any("skipped" in note for note in environment.notes)


def test_a_workspace_that_does_not_exist_still_reports_its_filesystem(
    tmp_path: Path,
) -> None:
    """Free space is asked about before the directory is created."""

    environment = detect_environment(workspace=tmp_path / "not" / "yet" / "made")
    workspace = next(disk for disk in environment.disks if disk.label == "workspace")

    assert workspace.free_bytes is not None


def test_every_probe_failing_still_yields_a_usable_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection is not allowed to be the reason a run cannot start."""

    monkeypatch.setattr(detect_module, "_read_text", lambda *_, **__: None)
    monkeypatch.setattr(detect_module, "_run_query", lambda _: None)
    monkeypatch.setattr(detect_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(detect_module.os, "cpu_count", lambda: None)

    environment = detect_environment()

    assert environment.plan.device == "cpu"
    assert environment.plan.recommended_workers >= 1


# ---------------------------------------------------------------------- cgroup


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("max 100000\n", None),
        ("200000 100000\n", 2.0),
        ("50000 100000\n", 0.5),
        ("garbage\n", None),
        ("100000\n", None),
    ],
)
def test_cgroup_v2_cpu_quota_is_read_as_a_core_count(
    monkeypatch: pytest.MonkeyPatch, contents: str, expected: float | None
) -> None:
    monkeypatch.setattr(
        detect_module,
        "_read_text",
        lambda path, **_: contents if str(path).endswith("cpu.max") else None,
    )

    assert detect_module._cgroup_cpu_quota() == expected


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("max\n", None),
        ("2147483648\n", 2147483648),
        ("0\n", None),
        # cgroup v1 spells "unlimited" as a number near 2**63, which is not a
        # memory ceiling any machine actually has.
        ("9223372036854771712\n", None),
    ],
)
def test_cgroup_memory_ceiling_ignores_the_unlimited_sentinels(
    monkeypatch: pytest.MonkeyPatch, contents: str, expected: int | None
) -> None:
    monkeypatch.setattr(
        detect_module,
        "_read_text",
        lambda path, **_: contents if str(path).endswith("memory.max") else None,
    )

    assert detect_module._cgroup_memory_limit() == expected


def test_a_container_limit_below_host_memory_is_the_one_that_counts() -> None:
    memory = MemoryInfo(
        total_bytes=512 * _GIB,
        available_bytes=400 * _GIB,
        cgroup_limit_bytes=16 * _GIB,
    )

    assert memory.effective_total_bytes == 16 * _GIB
    assert memory.total_gib == 16.0


def test_a_cgroup_ceiling_above_host_memory_does_not_invent_capacity() -> None:
    memory = MemoryInfo(total_bytes=8 * _GIB, cgroup_limit_bytes=512 * _GIB)

    assert memory.effective_total_bytes == 8 * _GIB


def test_a_cpu_quota_below_the_affinity_mask_caps_the_usable_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect_module.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(detect_module.os, "sched_getaffinity", lambda _: set(range(64)))
    monkeypatch.setattr(detect_module, "_cgroup_cpu_quota", lambda: 4.0)
    monkeypatch.setattr(detect_module, "_physical_cores_from_proc", lambda: 32)
    monkeypatch.setattr(detect_module, "_cpu_model_and_features", lambda: (None, ()))

    cpu = detect_module._detect_cpu()

    assert cpu.usable_cores == 4
    assert cpu.logical_cores == 64


# ------------------------------------------------------------------- /proc/cpu


def test_physical_cores_count_packages_and_cores_not_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four hyperthreads across two cores are two cores, not four."""

    cpuinfo = "\n".join(
        [
            "processor\t: 0",
            "physical id\t: 0",
            "core id\t\t: 0",
            "",
            "processor\t: 1",
            "physical id\t: 0",
            "core id\t\t: 0",
            "",
            "processor\t: 2",
            "physical id\t: 0",
            "core id\t\t: 1",
            "",
            "processor\t: 3",
            "physical id\t: 1",
            "core id\t\t: 1",
            "",
        ]
    )
    monkeypatch.setattr(detect_module, "_read_text", lambda *_, **__: cpuinfo)

    assert detect_module._physical_cores_from_proc() == 3


def test_a_cpuinfo_without_topology_reports_nothing_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detect_module, "_read_text", lambda *_, **__: "processor\t: 0\nmodel name\t: X\n"
    )

    assert detect_module._physical_cores_from_proc() is None


def test_only_instruction_sets_that_change_throughput_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detect_module,
        "_read_text",
        lambda *_, **__: (
            "model name\t: Fictional CPU\n"
            "flags\t\t: fpu vme avx avx2 fma sse4_2 constant_tsc rdrand nonstop_tsc\n"
        ),
    )

    model, features = detect_module._cpu_model_and_features()

    assert model == "Fictional CPU"
    assert features == ("avx", "avx2", "fma", "sse4_2")
    assert "rdrand" not in features


# ----------------------------------------------------------------- nvidia-smi


_EIGHT_A6000 = "\n".join(
    f"{index}, NVIDIA RTX A6000, 49140, 48887, 550.90.07, 8.6" for index in range(8)
)


def test_a_real_nvidia_smi_table_is_parsed_into_devices() -> None:
    devices = detect_module._parse_nvidia_csv(_EIGHT_A6000, has_compute_cap=True)

    assert len(devices) == 8
    assert devices[0].vendor is AcceleratorVendor.NVIDIA
    assert devices[0].name == "NVIDIA RTX A6000"
    assert devices[0].memory_total_mib == 49140
    assert devices[0].compute_capability == "8.6"
    assert devices[0].driver_version == "550.90.07"
    assert devices[7].index == 7


def test_a_single_4090_is_parsed_the_same_way() -> None:
    row = "0, NVIDIA GeForce RTX 4090, 24564, 24211, 555.42.02, 8.9"

    devices = detect_module._parse_nvidia_csv(row, has_compute_cap=True)

    assert len(devices) == 1
    assert devices[0].memory_total_gib == 23.99
    assert devices[0].compute_capability == "8.9"


def test_the_not_available_marker_is_not_treated_as_a_number() -> None:
    """``nounits`` strips the suffix but leaves ``[N/A]`` where unsupported."""

    row = "0, NVIDIA GeForce GTX 1080, [N/A], [N/A], 470.223.02, [N/A]"

    devices = detect_module._parse_nvidia_csv(row, has_compute_cap=True)

    assert devices[0].memory_total_mib is None
    assert devices[0].memory_free_mib is None
    assert devices[0].compute_capability is None
    assert devices[0].driver_version == "470.223.02"


def test_short_and_blank_rows_are_dropped_rather_than_half_parsed() -> None:
    text = "\n".join(
        [
            "",
            "0, NVIDIA RTX A6000, 49140",
            "not-a-number, NVIDIA RTX A6000, 49140, 48887, 550.90.07, 8.6",
            "1, NVIDIA RTX A6000, 49140, 48887, 550.90.07, 8.6",
        ]
    )

    devices = detect_module._parse_nvidia_csv(text, has_compute_cap=True)

    assert [device.index for device in devices] == [1]


def test_an_older_driver_without_compute_cap_still_yields_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full query is rejected wholesale by older drivers, not blanked."""

    calls: list[str] = []

    def fake_query(command: list[str]) -> str | None:
        calls.append(command[1])
        if "compute_cap" in command[1]:
            return None
        return "0, Tesla V100-SXM2-16GB, 16160, 16151, 418.87.00"

    monkeypatch.setattr(detect_module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(detect_module, "_run_query", fake_query)

    devices = detect_module._detect_nvidia([])

    assert len(calls) == 2
    assert len(devices) == 1
    assert devices[0].compute_capability is None
    assert devices[0].name == "Tesla V100-SXM2-16GB"


def test_an_nvidia_smi_that_answers_nothing_is_recorded_as_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect_module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(detect_module, "_run_query", lambda _: "Unable to determine\n")
    notes: list[str] = []

    assert detect_module._detect_nvidia(notes) == []
    assert notes and "no usable device rows" in notes[0]


def test_no_nvidia_smi_means_no_subprocess_is_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect_module.shutil, "which", lambda _: None)

    def explode(_: list[str]) -> str | None:  # pragma: no cover - must not run
        raise AssertionError("a query was issued for a tool that is not installed")

    monkeypatch.setattr(detect_module, "_run_query", explode)

    assert detect_module._detect_nvidia([]) == []


# ------------------------------------------------------------------- planning


def _cpu(cores: int) -> CpuInfo:
    return CpuInfo(logical_cores=cores, usable_cores=cores)


def test_one_core_is_left_for_everything_that_is_not_a_worker() -> None:
    plan = detect_module._plan(_cpu(16), MemoryInfo(total_bytes=64 * _GIB), ())

    assert plan.recommended_workers == 15


def test_a_single_core_machine_still_gets_one_worker() -> None:
    plan = detect_module._plan(_cpu(1), MemoryInfo(total_bytes=4 * _GIB), ())

    assert plan.recommended_workers == 1


def test_worker_count_is_capped_so_a_192_core_node_does_not_fork_192_times() -> None:
    plan = detect_module._plan(_cpu(192), MemoryInfo(total_bytes=1024 * _GIB), ())

    assert plan.recommended_workers == 32


@pytest.mark.parametrize(
    ("gib", "batch"),
    [(4, 8_192), (12, 16_384), (32, 65_536), (512, 131_072)],
)
def test_batch_size_follows_available_memory(gib: int, batch: int) -> None:
    plan = detect_module._plan(_cpu(8), MemoryInfo(total_bytes=gib * _GIB), ())

    assert plan.recommended_batch_size == batch


def test_unknown_memory_produces_a_conservative_batch_not_the_largest() -> None:
    plan = detect_module._plan(_cpu(8), MemoryInfo(), ())

    assert plan.recommended_batch_size == 32_768
    assert "unknown" in plan.rationale


def test_a_detected_nvidia_device_selects_cuda_zero() -> None:
    gpu = GpuDevice(
        index=0,
        vendor=AcceleratorVendor.NVIDIA,
        name="NVIDIA GeForce RTX 4090",
        memory_total_mib=24564,
    )

    plan = detect_module._plan(_cpu(32), MemoryInfo(total_bytes=128 * _GIB), (gpu,))

    assert plan.device == "cuda:0"
    assert plan.gpu_count == 1
    assert plan.gpu_backends_runnable is True
    assert "RTX 4090" in plan.rationale


def test_apple_silicon_selects_mps_rather_than_cuda() -> None:
    gpu = GpuDevice(index=0, vendor=AcceleratorVendor.APPLE, name="Apple silicon GPU")

    plan = detect_module._plan(_cpu(10), MemoryInfo(total_bytes=32 * _GIB), (gpu,))

    assert plan.device == "mps"


def test_no_accelerator_disables_gpu_backends_explicitly() -> None:
    plan = detect_module._plan(_cpu(8), MemoryInfo(total_bytes=16 * _GIB), ())

    assert plan.device == "cpu"
    assert plan.gpu_backends_runnable is False


def test_total_accelerator_memory_sums_across_a_multi_gpu_node() -> None:
    gpus = tuple(
        GpuDevice(
            index=index,
            vendor=AcceleratorVendor.NVIDIA,
            name="NVIDIA RTX A6000",
            memory_total_mib=49140,
        )
        for index in range(8)
    )
    environment = detect_environment(include_gpus=False).model_copy(update={"gpus": gpus})

    assert environment.has_gpu is True
    assert environment.total_gpu_memory_gib == pytest.approx(383.91, abs=0.01)


# ----------------------------------------------------------------- frameworks


def _installed(
    monkeypatch: pytest.MonkeyPatch, distributions: dict[str, str]
) -> None:
    """Pretend exactly these distributions are installed, and nothing else."""

    def version(name: str) -> str:
        try:
            return distributions[name]
        except KeyError:
            raise detect_module.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(detect_module.metadata, "version", version)


def test_frameworks_are_identified_without_importing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing torch to ask whether torch is installed is not acceptable."""

    import sys

    # ``None`` in sys.modules makes any ``import torch`` raise ImportError, so a
    # probe that reached for the module rather than its metadata would fail here.
    monkeypatch.setitem(sys.modules, "torch", None)
    _installed(monkeypatch, {"torch": "2.9.1+cu128", "onnxruntime": "1.29.0"})

    frameworks = {item.distribution: item for item in detect_module._detect_frameworks()}

    assert frameworks["torch"].build == "cu128"
    assert frameworks["torch"].accelerated is True
    assert frameworks["onnxruntime"].accelerated is False


def test_a_cpu_only_torch_build_is_not_reported_as_accelerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed(monkeypatch, {"torch": "2.9.1+cpu"})

    frameworks = detect_module._detect_frameworks()

    assert len(frameworks) == 1
    assert frameworks[0].accelerated is False
    assert frameworks[0].build == "cpu"


def test_a_rocm_torch_build_is_accelerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA is not the only local suffix that means a GPU wheel."""

    _installed(monkeypatch, {"torch": "2.9.1+rocm6.4"})

    assert detect_module._detect_frameworks()[0].accelerated is True


def test_the_gpu_onnxruntime_distribution_is_accelerated_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed(monkeypatch, {"onnxruntime-gpu": "1.29.0"})

    frameworks = detect_module._detect_frameworks()

    assert frameworks[0].distribution == "onnxruntime-gpu"
    assert frameworks[0].accelerated is True
