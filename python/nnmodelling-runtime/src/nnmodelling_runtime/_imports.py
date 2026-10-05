from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType


def resource_imports(runtime_package: str, resource_package: str, resource_dir: Path) -> dict[str, object]:
    """Load resource imports under private runtime names, including relative helpers."""
    abi_root = f"{runtime_package}._stereotype_runtime"
    if abi_root not in sys.modules:
        runtime_dir = Path(sys.modules[runtime_package].__file__).resolve().parent
        abi_dir = runtime_dir.parent / "stereotype_runtime"
        spec = importlib.util.spec_from_file_location(abi_root, abi_dir / "__init__.py", submodule_search_locations=[str(abi_dir)])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load private stereotype ABI from {abi_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[abi_root] = module
        spec.loader.exec_module(module)

    original_import = builtins.__import__

    def load_relative(full_name: str) -> ModuleType:
        if full_name in sys.modules:
            return sys.modules[full_name]
        relative = full_name[len(resource_package):].lstrip(".").split(".")
        module_file = resource_dir.joinpath(*relative).with_suffix(".py")
        package_file = resource_dir.joinpath(*relative, "__init__.py")
        is_package = package_file.is_file()
        source = package_file if is_package else module_file
        if not source.is_file() or not source.resolve().is_relative_to(resource_dir.resolve()):
            raise ModuleNotFoundError(f"resource helper {full_name!r} not found")
        spec = importlib.util.spec_from_file_location(full_name, source, submodule_search_locations=[str(source.parent)] if is_package else None)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load resource helper {source}")
        module = importlib.util.module_from_spec(spec)
        module.__builtins__ = make_builtins()
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        parent_name, _, child_name = full_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
        return module

    def make_builtins():
        def private_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "stereotype_runtime" or name.startswith("stereotype_runtime.")):
                private_name = abi_root + name[len("stereotype_runtime"):]
                loaded = importlib.import_module(private_name)
                return loaded if fromlist else sys.modules[abi_root]
            if level == 0 and (name == "nnmodelling_runtime" or name.startswith("nnmodelling_runtime.")):
                private_name = runtime_package + name[len("nnmodelling_runtime"):]
                loaded = importlib.import_module(private_name)
                return loaded if fromlist else sys.modules[runtime_package]
            if level and globals and globals.get("__package__", "").startswith(resource_package):
                base = globals["__package__"]
                absolute = importlib.util.resolve_name("." * level + name, base)
                if absolute == resource_package or absolute.startswith(resource_package + "."):
                    loaded = sys.modules[resource_package] if absolute == resource_package else load_relative(absolute)
                    if absolute == resource_package:
                        for child in fromlist:
                            if child != "*" and not hasattr(loaded, child):
                                load_relative(resource_package + "." + child)
                    return loaded if fromlist else sys.modules[resource_package]
            if level == 0 and "." not in name:
                candidate = resource_dir / f"{name}.py"
                package_candidate = resource_dir / name / "__init__.py"
                if candidate.is_file() or package_candidate.is_file():
                    loaded = load_relative(resource_package + "." + name)
                    return loaded if fromlist else sys.modules[resource_package]
            return original_import(name, globals, locals, fromlist, level)
        return {**vars(builtins), "__import__": private_import}

    return make_builtins()


def resource_module(module_name: str, directory: Path, entrypoint: Path, runtime_package: str) -> ModuleType:
    package_name = module_name + "_" + uuid.uuid4().hex
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    full_name = f"{package_name}.{entrypoint.stem}"
    spec = importlib.util.spec_from_file_location(full_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load resource module {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    module.__builtins__ = resource_imports(runtime_package, package_name, directory)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        sys.modules.pop(package_name, None)
        raise
    return module
