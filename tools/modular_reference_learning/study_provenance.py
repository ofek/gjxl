"""Freeze the research executable, build inputs and generated hooks for new studies."""
import shutil
import subprocess
from pathlib import Path

from analyze import sha


def freeze_build(binary, output):
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    artifacts = {}

    def copy(path, target):
        if target.exists():
            raise ValueError(f"Refusing to replace a frozen artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        artifacts[str(target)] = sha(target)

    frozen = output / "reference_learning"
    copy(binary, frozen)
    paths = [p for p in source.iterdir() if p.suffix in (".cpp", ".h", ".py") or p.name == "CMakeLists.txt"]
    paths += [repo / "CMakeLists.txt", repo / "LICENSE"]
    # Bind the native sources behind the linked archives as well as the small
    # directly compiled fitter. New private adapters can change other native
    # modules; archive hashes alone do not preserve their source provenance.
    for directory in ("src", "include", "cmake"):
        paths += [p for p in (repo / directory).rglob("*") if p.is_file()]
    paths += list((repo / "tests").glob("modular*"))
    paths += list((repo / "tests").glob("ans*"))
    paths += list((repo / "tests").glob("entropy*"))
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        copy(path, output / "source" / path.relative_to(repo))
    patch = output / "working-tree.patch"
    patch.write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"]))
    artifacts[str(patch)] = sha(patch)
    for name in ("CMakeCache.txt", "compile_commands.json"):
        copy(binary.parent / name, output / name)
    if (binary.parent / 'modular_wp_cost_reference.cpp').exists():
        copy(binary.parent / 'modular_wp_cost_reference.cpp', output / 'modular_wp_cost_reference.cpp')
    cache = dict(line.split("=", 1) for line in (binary.parent / "CMakeCache.txt").read_text().splitlines()
                 if line and not line.startswith(("#", "//")) and "=" in line)
    roots = {key.split(":")[0]: Path(value) for key, value in cache.items() if key.split(":")[0] in
             ("GJXL_BUILD", "REFERENCE_BUILD", "GENERATED_SOURCE")}
    for path in sorted(roots["GENERATED_SOURCE"].rglob("*")):
        if path.is_file():
            copy(path, output / "generated" / path.relative_to(roots["GENERATED_SOURCE"]))
    for key, names in {
        "GJXL_BUILD": ("libgjxl_codestream.a", "libgjxl_codec.a", "libgjxl_gpu_ops.a", "libgjxl_gpu.a"),
        "REFERENCE_BUILD": ("lib/libjxl-internal.a", "third_party/highway/libhwy.a", "lib/libjxl_cms.dylib",
                            "third_party/brotli/libbrotlienc.dylib", "third_party/brotli/libbrotlidec.dylib",
                            "third_party/brotli/libbrotlicommon.dylib"),
    }.items():
        for name in names:
            path = roots[key] / name
            artifacts[str(path)] = sha(path)
    return frozen, {"artifacts": artifacts,
                    "source_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                    "source_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
                    "source_snapshot_is_authoritative": True}
