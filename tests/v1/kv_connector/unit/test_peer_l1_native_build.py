# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native cache isolates toolchains and survives interrupted first builds."""

import os
import select
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import native

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tiny_build(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "owner.cpp").write_text("int answer() { return 42; }\n")
    compiler = tmp_path / "compiler"
    compiler.write_text(
        "#!/bin/sh\n"
        'echo build >> "$BUILD_LOG"\n'
        '[ -z "$BUILD_ARGV0" ] || echo "$0" > "$BUILD_ARGV0"\n'
        'if [ -n "$BUILD_READY" ]; then\n'
        '  echo ready > "$BUILD_READY"\n'
        '  read ignored < "$BUILD_GATE"\n'
        "fi\n"
        'while [ "$1" != "-o" ]; do shift; done\n'
        'printf complete > "$2"\n'
    )
    compiler.chmod(0o755)
    monkeypatch.setenv("CXX", str(compiler))
    monkeypatch.setenv("BUILD_LOG", str(tmp_path / "builds"))
    return source, tmp_path / "cache", compiler


def test_native_cache_key_covers_build_environment(tiny_build, monkeypatch):
    source, cache, compiler = tiny_build
    targets = [native._build_native(source, cache)]
    targets.append(native._build_native(source, cache))
    assert targets[0] == targets.pop()
    compiler.write_text(compiler.read_text() + "\n# changed compiler\n")
    targets.append(native._build_native(source, cache))
    monkeypatch.setattr(native.platform, "machine", lambda: "another-architecture")
    targets.append(native._build_native(source, cache))
    monkeypatch.setattr(native.platform, "system", lambda: "another-os")
    targets.append(native._build_native(source, cache))
    monkeypatch.setattr(native, "_COMPILE_FLAGS", (*native._COMPILE_FLAGS, "-g"))
    targets.append(native._build_native(source, cache))
    (source / "owner.cpp").write_text("int answer() { return 43; }\n")
    targets.append(native._build_native(source, cache))
    assert len(set(targets)) == len(targets)
    assert all(target.read_bytes() == b"complete" for target in targets)


def test_compiler_runs_under_its_configured_name(tiny_build, monkeypatch):
    source, cache, compiler = tiny_build
    alias = compiler.parent / "c++"
    alias.symlink_to(compiler)
    monkeypatch.setenv("CXX", str(alias))
    monkeypatch.setenv("BUILD_ARGV0", str(source.parent / "argv0"))
    native._build_native(source, cache)
    assert (source.parent / "argv0").read_text().strip() == str(alias)


def test_concurrent_first_build_compiles_once(tiny_build):
    source, cache, _ = tiny_build
    with ThreadPoolExecutor(2) as pool:
        targets = list(
            pool.map(lambda _: native._build_native(source, cache), range(2))
        )
    assert targets[0] == targets[1]
    assert (source.parent / "builds").read_text().splitlines() == ["build"]


def test_killed_builder_does_not_strand_cache_lock(tiny_build, monkeypatch):
    source, cache, _ = tiny_build
    ready, gate = source.parent / "ready", source.parent / "gate"
    os.mkfifo(ready)
    os.mkfifo(gate)
    script = (
        "from pathlib import Path; "
        "from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native "
        "import _build_native; "
        f"_build_native(Path({str(source)!r}), Path({str(cache)!r}))"
    )
    env = {**os.environ, "BUILD_READY": str(ready), "BUILD_GATE": str(gate)}
    process = subprocess.Popen(
        [sys.executable, "-c", script], env=env, start_new_session=True
    )
    descriptor = os.open(ready, os.O_RDONLY | os.O_NONBLOCK)
    try:
        assert select.select([descriptor], [], [], 60)[0]
        assert os.read(descriptor, 100).strip() == b"ready"
    finally:
        os.close(descriptor)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    result = subprocess.run([sys.executable, "-c", script], timeout=60, check=True)
    assert result.returncode == 0
    assert next(cache.glob("*/owner.so")).read_bytes() == b"complete"
