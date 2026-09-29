"""Run the shared NIRBENCH-DL protocol on a model supplied by the user.

Place the model file in ``models/newmodel/``. The selected builder must accept
one positional argument (the number of spectral features) and return an
uncompiled Keras model with one continuous output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

ROOT = Path(__file__).resolve().parent
NEW_MODEL_DIR = ROOT / "models" / "newmodel"
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def available_model_files() -> list[Path]:
    """Return user model files in a stable order."""
    return sorted(
        path
        for path in NEW_MODEL_DIR.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def resolve_model_file(value: str) -> Path:
    """Resolve a filename while preventing reads outside models/newmodel/."""
    supplied = Path(value)
    if supplied.suffix == "":
        supplied = supplied.with_suffix(".py")
    candidate = supplied if supplied.is_absolute() else NEW_MODEL_DIR / supplied
    candidate = candidate.resolve()
    model_root = NEW_MODEL_DIR.resolve()
    try:
        relative = candidate.relative_to(model_root)
    except ValueError as exc:
        raise ValueError(f"Model file must be inside {model_root}") from exc
    if len(relative.parts) != 1:
        raise ValueError("Model file must be directly inside models/newmodel/")
    if candidate.suffix.lower() != ".py" or candidate.name.startswith("_"):
        raise ValueError("Model file must be a public .py file")
    if not candidate.is_file():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def load_module(path: Path) -> ModuleType:
    """Import one user model under an isolated module name."""
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"nirbench_external_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def select_builder(module: ModuleType, requested_name: str | None) -> tuple[str, Callable]:
    """Select an explicit builder, build_model, or one unambiguous model factory."""
    if requested_name:
        builder = getattr(module, requested_name, None)
        if not callable(builder):
            raise ValueError(
                f"Builder {requested_name!r} is not a callable in {module.__file__}"
            )
        name = requested_name
    else:
        default = getattr(module, "build_model", None)
        if callable(default):
            name, builder = "build_model", default
        else:
            candidates = [
                (name, value)
                for name, value in vars(module).items()
                if callable(value)
                and getattr(value, "__module__", None) == module.__name__
                and not name.startswith("_")
                and (name.startswith("build_") or name.endswith("_model"))
            ]
            if len(candidates) != 1:
                names = [name for name, _ in candidates]
                raise ValueError(
                    "Could not select one model builder automatically. "
                    "Define build_model(...) or pass --builder FUNCTION. "
                    f"Candidates: {names or 'none'}"
                )
            name, builder = candidates[0]

    try:
        inspect.signature(builder).bind(100)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Builder {name!r} must accept the spectral feature count as its "
            "only required positional argument; other arguments need defaults."
        ) from exc
    return name, builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Keras model from models/newmodel/.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-file",
        help=(
            "Python filename in models/newmodel/ (the .py suffix is optional). "
            "If omitted, the only model file in that folder is selected."
        ),
    )
    parser.add_argument(
        "--builder",
        help=(
            "Builder function in the model file. Default: build_model; if absent,\n"
            "one unambiguous build_* or *_model function is selected."
        ),
    )
    parser.add_argument(
        "--name",
        help="Output model name. Default: model filename without .py.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="Dataset folder name(s). Default: all 30 datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List model files currently available in models/newmodel/ and exit.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        files = available_model_files()
        if files:
            print("Available external models:")
            for path in files:
                print(f"  {path.name}")
        else:
            print(f"No model files found in {NEW_MODEL_DIR}")
        return

    try:
        if args.model_file:
            model_file = resolve_model_file(args.model_file)
        else:
            files = available_model_files()
            if len(files) != 1:
                names = [path.name for path in files]
                raise ValueError(
                    "Omit --model-file only when models/newmodel/ contains exactly "
                    f"one model file; found {names or 'none'}"
                )
            model_file = files[0]
        model_name = args.name or model_file.stem
        if not MODEL_NAME_PATTERN.fullmatch(model_name):
            raise ValueError(
                "--name may contain letters, numbers, periods, underscores, and hyphens; "
                "it must start with a letter or number"
            )

        # Importing benchmark first establishes the same TensorFlow runtime and seeds
        # used by the built-in architectures before user code imports TensorFlow.
        from benchmark import (
            _run_standard_cnn_baseline_all_datasets,
            build_global_model_ranking,
            keras,
        )

        module = load_module(model_file)
        builder_name, user_builder = select_builder(module, args.builder)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    def build_adapter(input_vector_dimension: int):
        model = user_builder(input_vector_dimension)
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"{builder_name} returned {type(model).__name__}; expected a Keras Model"
            )
        input_shape = model.input_shape
        expected_input_shape = (input_vector_dimension, 1)
        if isinstance(input_shape, list) or tuple(input_shape[1:]) != expected_input_shape:
            raise ValueError(
                "The external model must have one input with shape "
                f"{expected_input_shape}; got {input_shape}"
            )
        output_shape = model.output_shape
        if isinstance(output_shape, list) or output_shape[-1] != 1:
            raise ValueError(
                f"The external model must have one continuous output; got {output_shape}"
            )
        return model

    print(f"External model file: {model_file.relative_to(ROOT)}")
    print(f"Builder: {builder_name}")
    print(f"Result name: {model_name}")
    print(f"Datasets: {args.dataset or 'all'}")

    _run_standard_cnn_baseline_all_datasets(
        model_name=model_name,
        build_model_fn=build_adapter,
        dataset_filter=args.dataset,
        batch_size=args.batch_size,
    )
    build_global_model_ranking()


if __name__ == "__main__":
    main()
