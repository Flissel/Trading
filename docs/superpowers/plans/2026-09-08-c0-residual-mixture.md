# C.0 Residual Mixture with GRU Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the torch-optional C.0 subsystem of the CTM mixture spec — causal slope and spectral view features, a regime-gated residual mixture over four shallow experts with a GRU sequence encoder, a leakage-safe registered fold runner with compute evidence, and the `base` / `best_expert` / `uniform_moe` / `gru_moe` comparator configurations — so that Plan 2 can plug a faithful CTM encoder into the same encoder contract.

**Architecture:** View features are Decimal and live in the core package; everything that needs PyTorch lives in `trading_bot.neural` behind an actionable import guard. The mixture is `logit = f0 + Σ π_k r_k` over a frozen Phase-A logistic base `f0`, zero-initialized residual experts, and a soft gate produced by the encoder (RG-ResMoE recipe). The fold runner reuses the Phase-A verification, split, abstention, floor-selection, bootstrap, and immutable-write helpers and records outcomes in a new `neural_fold_runs` registry table.

**Tech Stack:** Python 3.12, Pydantic 2, DuckDB, PyArrow, PyTorch ≥ 2.7 (CUDA 12.6 wheel on Windows, optional `neural` dependency group), pytest, Ruff, Mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-08-ctm-moe-challenger-design.md` (sections 4, 6, 7, 8, 9, 10 step C.0). Plan 2 (`C.1/C.2`) covers the vendored CTM encoder, `ctm_alone`, `ctm_moe`, and the decision artifact; it is written after this plan's encoder contract exists.

## Global Constraints

- Base branch: `codex/phase-a-edge-recovery` (worktree `.worktrees/phase-a-edge-recovery`). Before Task 1, cherry-pick the spec and plan docs commits from `codex/phase1-foundation` onto it (`git cherry-pick daad19f <plan-commit>`), docs only.
- Python `>=3.12,<3.13`; mypy `strict`; ruff `line-length = 100`, rules `E, F, I, UP, B, SIM, RUF`; never `Any`; no `console.log`-style debugging output; no floats in canonical JSON (`Decimal` everywhere an artifact is hashed; convert floats with `Decimal(str(value))`).
- The core package stays torch-free: `trading_bot.neural.spec` and `trading_bot.neural._torch` import no torch; `trading_bot.view_features` imports no torch. Torch modules fail closed with `NeuralDependencyError` carrying the hint `uv sync --group neural`.
- Frozen numbers from the spec: window 96 bars, block length 96, 2,000 bootstrap repetitions, minimum 200 selection trades, three seeds per cell, `d_model` 256, `iterations` 20, `memory_length` 16, batch 256, learning rate `0.001`, patience 5, at most 30 epochs, VRAM limit `12884901888` bytes. Tests may lower `max_epochs`, `block_length`, `bootstrap_repetitions`, and `minimum_selection_trades` only.
- Expert order is frozen: `("slope", "spectral", "rate", "summary")`. Encoder residual is gate entry index 4.
- Regime information reaches experts only through the gate weights. Expert inputs are the decision bar's own view columns; the encoder input is the trailing 96-bar window.
- The final holdout is never read; every fold document carries `final_holdout_status: "locked"` and `final_holdout_read_count: 0`.
- Every artifact is canonical-JSON hashed, written atomically, immutable (`OUTPUT_ALREADY_EXISTS` on collision), and never repaired or overwritten.
- The neural campaign uses its **own registry database file** (`artifacts/neural/hypotheses-neural.sqlite3` in production); it must never share the Phase-A registry, because `phase_a_decision._registry_census` selects families by audit and schema hash and would count the neural family as an incomplete Phase-A campaign.
- Determinism: `seed_everything` sets Python and torch seeds, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; inference latency is measured on CPU.
- Test environment on this machine: `TEMP`/`TMP` point to `E:\Temp` and the storage policy excludes `E:` by design. Run every pytest command with `$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP`.
- No real-capital mode, live endpoint, order route, or promotion authority is added. `no_edge_found` is a valid result. Nothing in this plan registers an economic decision.
- Scope boundaries against the spec: section 6.7 (real-time shadow path) belongs to Phase B / P1.24 and is not built here; the spec's `run_ctm_mixture_fold` / `verify_ctm_mixture_fold` are implemented as `run_neural_fold` / `verify_neural_fold` with `encoder_kind` selecting the encoder, so one runner serves C.0, C.1, and C.2.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | optional `neural` dependency group, CUDA index, mypy override |
| `src/trading_bot/neural/__init__.py` | package docstring only; never imports torch |
| `src/trading_bot/neural/_torch.py` | `NeuralDependencyError`, `INSTALL_HINT`, `seed_everything` |
| `src/trading_bot/feature_matrix.py` | existing; gains generic `ColumnStatistics`, `fit_column_statistics`, `standardize_row` |
| `src/trading_bot/view_features.py` | causal slope (E1) and spectral (E2) features, `ViewSample`, view matrix fit/transform |
| `src/trading_bot/hypothesis_registry.py` | existing; family name widened, `NeuralFoldRun` + `neural_fold_runs` table |
| `src/trading_bot/phase_a_fold_run.py` | existing; `verify_research_inputs` extracted, `FAMILY_NOT_PHASE_A` fail-closed |
| `src/trading_bot/neural/spec.py` | `CtmMixtureSpec`, family, campaign hash, registration, spec loader (torch-free) |
| `src/trading_bot/neural/windows.py` | column layout, row tensor, membership positions, window gather |
| `src/trading_bot/neural/mixture.py` | residual experts, GRU encoder, `ResidualMixture`, certainty, loss |
| `src/trading_bot/neural/training.py` | deterministic training loop with early stopping |
| `src/trading_bot/neural/compute_evidence.py` | parameter count, peak VRAM, latency percentiles |
| `src/trading_bot/neural/fold_run.py` | `NeuralFoldConfig`, `run_neural_fold`, `verify_neural_fold` |
| `src/trading_bot/cli.py` | existing; `neural-register-campaign`, `neural-fold` |
| `README.md` | existing; C.0 section |
| `tests/test_neural_boundary.py` | torch-free tests of the import guard |
| `tests/test_view_features.py` | view feature tests |
| `tests/neural/__init__.py`, `tests/neural/conftest.py` | torch-gated test package (`pytest.importorskip("torch")`) |
| `tests/neural/test_spec.py`, `test_windows.py`, `test_mixture.py`, `test_training.py`, `test_compute_evidence.py`, `test_fold_run.py`, `test_cli.py` | one module per source module |

---

### Task 0: Branch preparation

**Files:** none modified.

- [ ] **Step 1: Bring the docs onto the implementation branch**

Run from the worktree:

```powershell
cd C:\Users\User\Documents\ChatGPT\Trading\.worktrees\phase-a-edge-recovery
git cherry-pick daad19f
git log --oneline -3
```

Expected: `docs: design ctm mixture-of-experts challenger` is the tip. Cherry-pick this plan's own commit from `codex/phase1-foundation` the same way once it exists (`git log codex/phase1-foundation --oneline -1`).

- [ ] **Step 2: Confirm the suite is green before any change**

```powershell
$env:TEMP='C:\Users\User\AppData\Local\Temp'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
uv sync
uv run pytest -q -p no:cacheprovider
```

Expected: `511 passed`.

---

### Task 1: Optional `neural` dependency group and torch boundary

**Files:**
- Modify: `pyproject.toml`
- Create: `src/trading_bot/neural/__init__.py`
- Create: `src/trading_bot/neural/_torch.py`
- Create: `tests/test_neural_boundary.py`
- Create: `tests/neural/__init__.py`
- Create: `tests/neural/conftest.py`
- Create: `tests/neural/test_boundary.py`

**Interfaces:**
- Produces: `trading_bot.neural._torch.NeuralDependencyError(RuntimeError)`, `INSTALL_HINT: str`, `seed_everything(seed: int) -> None`.
- Later torch modules use this exact import pattern at the top of the file:

```python
from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error
```

- [ ] **Step 1: Write the failing torch-free tests**

`tests/test_neural_boundary.py`:

```python
import sys
from importlib import import_module

import pytest

from trading_bot.neural import _torch as boundary


def test_seed_everything_rejects_negative_or_bool_seed() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        boundary.seed_everything(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        boundary.seed_everything(True)


def test_seed_everything_fails_closed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(boundary.NeuralDependencyError, match="uv sync --group neural"):
        boundary.seed_everything(3)


def test_neural_package_import_is_torch_free() -> None:
    package = import_module("trading_bot.neural")
    assert not hasattr(package, "torch")
```

`tests/neural/conftest.py`:

```python
import pytest

pytest.importorskip("torch")
```

`tests/neural/test_boundary.py`:

```python
import torch

from trading_bot.neural._torch import seed_everything


def test_seed_everything_makes_torch_draws_reproducible() -> None:
    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)
    second = torch.rand(4)

    assert torch.equal(first, second)
    assert torch.are_deterministic_algorithms_enabled()
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_neural_boundary.py tests/neural -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.neural'`.

- [ ] **Step 3: Add the dependency group and the boundary module**

Append to `pyproject.toml` (keep every existing section):

```toml
[dependency-groups]
dev = [
  "mypy>=1.17,<2",
  "pytest>=8.4,<9",
  "pytest-cov>=6.2,<7",
  "ruff>=0.12,<1",
]
neural = [
  "torch>=2.7,<3",
]

[tool.uv]
default-groups = ["dev"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

`src/trading_bot/neural/__init__.py`:

```python
"""Optional neural challengers. Importing this package never imports PyTorch."""
```

`src/trading_bot/neural/_torch.py`:

```python
"""PyTorch boundary: actionable import guard and deterministic seeding."""

import os
import random

INSTALL_HINT = (
    "PyTorch is not installed; run `uv sync --group neural` to enable trading_bot.neural"
)


class NeuralDependencyError(RuntimeError):
    """Raised when the optional neural dependency group is unavailable."""


def seed_everything(seed: int) -> None:
    """Seed Python and torch and force deterministic kernels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:
        raise NeuralDependencyError(INSTALL_HINT) from error
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

`tests/neural/__init__.py` is empty.

- [ ] **Step 4: Install the group and run the tests**

```powershell
uv sync --group neural
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest tests/test_neural_boundary.py tests/neural -q -p no:cacheprovider
```

Expected: torch version printed (CUDA `True` on the RTX 3060; `False` is acceptable for tests), then `4 passed`.

- [ ] **Step 5: Typecheck and lint**

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
```

Expected: clean. If mypy reports errors *inside* torch's own stubs (not in our files), add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["torch", "torch.*"]
follow_imports = "silent"
```

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml uv.lock src/trading_bot/neural tests/test_neural_boundary.py tests/neural
git commit -m "feat: add optional neural dependency boundary"
```

---

### Task 2: Generic column statistics in `feature_matrix.py`

**Files:**
- Modify: `src/trading_bot/feature_matrix.py`
- Modify: `tests/test_feature_matrix.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class ColumnStatistics:
    medians: tuple[Decimal, ...]
    means: tuple[Decimal, ...]
    scales: tuple[Decimal, ...]

def fit_column_statistics(
    columns: tuple[tuple[Decimal | None, ...], ...], *, labels: tuple[str, ...]
) -> ColumnStatistics: ...

def standardize_row(
    statistics: ColumnStatistics, values: tuple[Decimal | None, ...]
) -> tuple[Decimal, ...]: ...   # (standardized, missing_indicator) pairs, flattened
```

- `FittedFeatureMatrix`, `fit_feature_matrix`, `transform_feature_matrix`, and `FeatureMatrixSpec.schema_hash` keep their exact shapes and formulas (Phase-A artifacts embed them).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feature_matrix.py`:

```python
from trading_bot.feature_matrix import (
    ColumnStatistics,
    fit_column_statistics,
    standardize_row,
)


def test_column_statistics_reproduce_feature_matrix_path() -> None:
    samples = training_samples()
    fields = ("return_4", "basis_change_4")
    fitted = fit_feature_matrix(samples, fields=fields)

    columns = tuple(
        tuple(getattr(sample, field) for sample in samples) for field in fields
    )
    statistics = fit_column_statistics(columns, labels=fields)

    assert statistics == ColumnStatistics(fitted.medians, fitted.means, fitted.scales)
    assert standardize_row(statistics, (Decimal(1000), Decimal(1000))) == (
        transform_feature_matrix(fitted, extreme_test_samples()).rows[0]
    )


def test_column_statistics_reject_all_missing_and_bad_widths() -> None:
    with pytest.raises(ValueError, match="all missing"):
        fit_column_statistics(((None, None),), labels=("basis_mean_16",))
    with pytest.raises(ValueError, match="aligned"):
        fit_column_statistics(((Decimal(1),),), labels=("a", "b"))
    statistics = fit_column_statistics(((Decimal(1), Decimal(3)),), labels=("a",))
    with pytest.raises(ValueError, match="width"):
        standardize_row(statistics, (Decimal(1), Decimal(2)))
    with pytest.raises(ValueError, match="positive"):
        ColumnStatistics((Decimal(1),), (Decimal(1),), (Decimal(0),))
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_feature_matrix.py -q -p no:cacheprovider
```

Expected: FAIL with `ImportError: cannot import name 'ColumnStatistics'`.

- [ ] **Step 3: Refactor `feature_matrix.py`**

Insert after `TransformedFeatureMatrix`:

```python
@dataclass(frozen=True, slots=True)
class ColumnStatistics:
    """Train-only imputation and standardization statistics for ordered columns."""

    medians: tuple[Decimal, ...]
    means: tuple[Decimal, ...]
    scales: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        width = len(self.medians)
        if not (len(self.means) == len(self.scales) == width):
            raise ValueError("column statistics have inconsistent widths")
        _validate_finite_statistics(*self.medians, *self.means, *self.scales)
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("column scales must be positive")


def fit_column_statistics(
    columns: tuple[tuple[Decimal | None, ...], ...], *, labels: tuple[str, ...]
) -> ColumnStatistics:
    """Fit median imputation and population standardization per ordered column."""
    if not columns or len(columns) != len(labels):
        raise ValueError("columns and labels must be non-empty and aligned")
    medians: list[Decimal] = []
    means: list[Decimal] = []
    scales: list[Decimal] = []
    for label, column in zip(labels, columns, strict=True):
        if not column:
            raise ValueError("column statistics require at least one sample")
        present = tuple(value for value in column if value is not None)
        if not present:
            raise ValueError(f"feature field {label!r} is all missing in training data")
        median = _median(present)
        imputed = tuple(median if value is None else value for value in column)
        count = Decimal(len(imputed))
        mean = sum(imputed, Decimal(0)) / count
        variance = sum(((value - mean) ** 2 for value in imputed), Decimal(0)) / count
        scale = variance.sqrt() if variance > 0 else Decimal(1)
        _validate_finite_statistics(median, mean, scale)
        medians.append(median)
        means.append(mean)
        scales.append(scale)
    return ColumnStatistics(tuple(medians), tuple(means), tuple(scales))


def standardize_row(
    statistics: ColumnStatistics, values: tuple[Decimal | None, ...]
) -> tuple[Decimal, ...]:
    """Impute, standardize, and append a missing indicator for every column."""
    if len(values) != len(statistics.medians):
        raise ValueError("row width does not match the column statistics")
    row: list[Decimal] = []
    for index, value in enumerate(values):
        missing = value is None
        imputed = statistics.medians[index] if value is None else value
        standardized = (imputed - statistics.means[index]) / statistics.scales[index]
        if not standardized.is_finite():
            raise ValueError("transformed feature values must be finite")
        row.extend((standardized, Decimal(1) if missing else Decimal(0)))
    return tuple(row)
```

Replace the body of `fit_feature_matrix` after the two validations with:

```python
    columns = tuple(
        tuple(_feature_value(sample, field) for sample in samples) for field in fields
    )
    statistics = fit_column_statistics(columns, labels=fields)
    spec = FeatureMatrixSpec(schema.schema_hash, fields)
    return FittedFeatureMatrix(
        fields=fields,
        medians=statistics.medians,
        means=statistics.means,
        scales=statistics.scales,
        schema_hash=spec.schema_hash,
    )
```

Replace the body of `transform_feature_matrix` after `_validate_unique_sample_ids(samples)` with:

```python
    statistics = ColumnStatistics(fitted.medians, fitted.means, fitted.scales)
    rows = tuple(
        standardize_row(
            statistics, tuple(_feature_value(sample, field) for field in fitted.fields)
        )
        for sample in samples
    )
    return TransformedFeatureMatrix(
        sample_ids=tuple(sample.base.sample_id for sample in samples),
        rows=rows,
    )
```

Keep `_validate_fitted` unchanged so its existing error messages survive.

- [ ] **Step 4: Run the full suite (regression guard for hash-bound artifacts)**

```powershell
uv run pytest tests/test_feature_matrix.py tests/test_phase_a_fold_run.py tests/test_phase_a_pipeline.py -q -p no:cacheprovider
```

Expected: all pass, including the new two.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/feature_matrix.py tests/test_feature_matrix.py
git commit -m "refactor: expose reusable column statistics"
```

---

### Task 3: View features I — schema, log series, slope family

**Files:**
- Create: `src/trading_bot/view_features.py`
- Create: `tests/test_view_features.py`

**Interfaces:**
- Produces:

```python
SLOPE_FIELDS = (
    "slope_ols_8", "slope_ols_32", "slope_ols_96",
    "slope_theil_sen_8", "slope_theil_sen_32",
    "slope_change_8", "slope_agreement",
)
SPECTRAL_FIELDS = (
    "acf_lag32_192", "acf_lag96_384", "acf_lag672_1344",
    "funding_phase_sin", "funding_phase_cos", "ljung_box_96", "envelope_width_96",
)
VIEW_FIELDS = SLOPE_FIELDS + SPECTRAL_FIELDS
VIEW_FORMULA_ID = "c0-view-features-v1"
FUNDING_PERIOD_NS = 28_800_000_000_000
EXTREMUM_CONFIRMATION_BARS = 8

class ViewFeatureSchema: formula_id, field_names, schema_hash
def c0_view_feature_schema() -> ViewFeatureSchema
class LogSeries: bars, log_close, log_return, index_by_source_id
def build_log_series(bars: tuple[ResearchBar, ...]) -> LogSeries
type FeatureResult = tuple[Decimal, tuple[ResearchBar, ...]] | None
def ols_slope(series, index, *, window, decision_time_ns) -> FeatureResult
def theil_sen_slope(series, index, *, window, decision_time_ns) -> FeatureResult
def slope_change(series, index, *, window, decision_time_ns) -> FeatureResult
def slope_agreement(*slopes: Decimal | None) -> Decimal | None
```

- Deviation from the spec table, recorded here: Theil–Sen is computed at 8 and 32 bars only. At 96 bars it needs 4,560 pairwise Decimal slopes per sample (≈ 250 million operations over the 580-day capture); OLS covers the 96-bar scale.

- [ ] **Step 1: Write the failing tests**

`tests/test_view_features.py`:

```python
from decimal import Decimal

import pytest

from trading_bot.bar_research import ResearchBar
from trading_bot.view_features import (
    VIEW_FIELDS,
    ViewFeatureSchema,
    build_log_series,
    c0_view_feature_schema,
    ols_slope,
    slope_agreement,
    slope_change,
    theil_sen_slope,
)

INTERVAL_NS = 900_000_000_000
GROWTH = Decimal("1.001")


def bar(index: int, close: Decimal, *, available_offset_ns: int = 1) -> ResearchBar:
    open_time_ns = index * INTERVAL_NS
    return ResearchBar(
        source_id=f"okx-{index}",
        venue="OKX",
        open_time_ns=open_time_ns,
        available_time_ns=open_time_ns + INTERVAL_NS + available_offset_ns,
        close=close,
        high=close * Decimal("1.01"),
        low=close * Decimal("0.99"),
        base_volume=Decimal(10),
    )


def exponential_bars(count: int) -> tuple[ResearchBar, ...]:
    return tuple(bar(index, Decimal(100) * GROWTH**index) for index in range(count))


def decision_time(series_index: int) -> int:
    return series_index * INTERVAL_NS + INTERVAL_NS + 1


def test_view_schema_binds_formula_and_ordered_fields() -> None:
    schema = c0_view_feature_schema()

    assert schema.field_names == VIEW_FIELDS
    assert len(schema.schema_hash) == 64
    with pytest.raises(ValueError, match="exact ordered fields"):
        ViewFeatureSchema("c0-view-features-v1", VIEW_FIELDS[::-1])


def test_ols_and_theil_sen_recover_log_growth_slope() -> None:
    series = build_log_series(exponential_bars(120))
    expected = GROWTH.ln()

    ols = ols_slope(series, 119, window=96, decision_time_ns=decision_time(119))
    theil = theil_sen_slope(series, 119, window=32, decision_time_ns=decision_time(119))

    assert ols is not None and theil is not None
    assert abs(ols[0] - expected) < Decimal("1e-20")
    assert abs(theil[0] - expected) < Decimal("1e-20")
    assert len(ols[1]) == 96
    assert ols[1][0].source_id == "okx-24"


def test_slopes_fail_closed_on_short_history_and_unavailable_bars() -> None:
    bars = list(exponential_bars(40))
    bars[30] = bar(30, bars[30].close, available_offset_ns=10**15)
    series = build_log_series(tuple(bars))

    assert ols_slope(series, 6, window=8, decision_time_ns=decision_time(6)) is None
    assert ols_slope(series, 39, window=32, decision_time_ns=decision_time(39)) is None
    assert ols_slope(series, 39, window=8, decision_time_ns=decision_time(39)) is not None


def test_slope_change_and_agreement() -> None:
    series = build_log_series(exponential_bars(120))
    change = slope_change(series, 119, window=8, decision_time_ns=decision_time(119))

    assert change is not None
    assert abs(change[0]) < Decimal("1e-20")
    assert len(change[1]) == 16
    assert slope_agreement(Decimal(1), Decimal(2), Decimal("0.5")) == Decimal(1)
    assert slope_agreement(Decimal(1), Decimal(-2), Decimal(0)) == Decimal(0)
    assert slope_agreement(Decimal(1), None, Decimal(1)) is None


def test_log_series_rejects_unordered_or_duplicate_bars() -> None:
    bars = exponential_bars(3)
    with pytest.raises(ValueError, match="chronological"):
        build_log_series((bars[1], bars[0]))
    with pytest.raises(ValueError, match="unique"):
        build_log_series((bars[0], bar(0, Decimal(101))))
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_view_features.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.view_features'`.

- [ ] **Step 3: Write the slope half of `view_features.py`**

```python
"""Causal slope and spectral view features for the C.0 residual mixture."""

import math
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from trading_bot.bar_research import ResearchBar
from trading_bot.canonical import content_sha256
from trading_bot.feature_matrix import (
    PREPROCESSING_FORMULA_ID,
    ColumnStatistics,
    fit_column_statistics,
    standardize_row,
)
from trading_bot.market_features import EnrichedBarSample

SLOPE_FIELDS = (
    "slope_ols_8",
    "slope_ols_32",
    "slope_ols_96",
    "slope_theil_sen_8",
    "slope_theil_sen_32",
    "slope_change_8",
    "slope_agreement",
)
SPECTRAL_FIELDS = (
    "acf_lag32_192",
    "acf_lag96_384",
    "acf_lag672_1344",
    "funding_phase_sin",
    "funding_phase_cos",
    "ljung_box_96",
    "envelope_width_96",
)
VIEW_FIELDS = SLOPE_FIELDS + SPECTRAL_FIELDS
VIEW_FORMULA_ID = "c0-view-features-v1"
FUNDING_PERIOD_NS = 28_800_000_000_000
EXTREMUM_CONFIRMATION_BARS = 8

type FeatureResult = tuple[Decimal, tuple[ResearchBar, ...]] | None


@dataclass(frozen=True, slots=True)
class ViewFeatureSchema:
    """Versioned identity for the fixed C.0 view feature contract."""

    formula_id: str
    field_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.formula_id:
            raise ValueError("formula identifier must not be empty")
        if self.field_names != VIEW_FIELDS:
            raise ValueError("view feature schema must use the exact ordered fields")

    @property
    def schema_hash(self) -> str:
        return content_sha256(
            {"formula_id": self.formula_id, "field_names": list(self.field_names)}
        )


def c0_view_feature_schema() -> ViewFeatureSchema:
    """Return the canonical public view schema for every C.0 integration."""
    return ViewFeatureSchema(formula_id=VIEW_FORMULA_ID, field_names=VIEW_FIELDS)


@dataclass(frozen=True, slots=True)
class LogSeries:
    """Bars with cached log closes and one-bar log returns (index zero return is unused)."""

    bars: tuple[ResearchBar, ...]
    log_close: tuple[Decimal, ...]
    log_return: tuple[Decimal, ...]
    index_by_source_id: dict[str, int]


def build_log_series(bars: tuple[ResearchBar, ...]) -> LogSeries:
    """Cache logarithms once so window features stay linear in the window length."""
    for previous, current in pairwise(bars):
        if current.open_time_ns <= previous.open_time_ns:
            raise ValueError("view feature bars must be strictly chronological")
    source_ids = tuple(bar.source_id for bar in bars)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("view feature bar source IDs must be unique")
    log_close = tuple(bar.close.ln() for bar in bars)
    log_return = (Decimal(0), *(later - earlier for earlier, later in pairwise(log_close)))
    return LogSeries(
        bars=bars,
        log_close=log_close,
        log_return=log_return,
        index_by_source_id={source_id: index for index, source_id in enumerate(source_ids)},
    )


def ols_slope(
    series: LogSeries, index: int, *, window: int, decision_time_ns: int
) -> FeatureResult:
    """Least-squares slope of log close against bar ordinal over the trailing window."""
    start = index - window + 1
    if start < 0:
        return None
    rows = series.bars[start : index + 1]
    if not _admissible(rows, decision_time_ns):
        return None
    values = series.log_close[start : index + 1]
    count = Decimal(window)
    x_mean = (count - Decimal(1)) / Decimal(2)
    y_mean = sum(values, Decimal(0)) / count
    numerator = sum(
        ((Decimal(position) - x_mean) * (value - y_mean) for position, value in enumerate(values)),
        Decimal(0),
    )
    denominator = sum(((Decimal(position) - x_mean) ** 2 for position in range(window)), Decimal(0))
    return (numerator / denominator, rows)


def theil_sen_slope(
    series: LogSeries, index: int, *, window: int, decision_time_ns: int
) -> FeatureResult:
    """Median of pairwise log-close slopes; robust to single-bar spikes."""
    start = index - window + 1
    if start < 0:
        return None
    rows = series.bars[start : index + 1]
    if not _admissible(rows, decision_time_ns):
        return None
    values = series.log_close[start : index + 1]
    slopes = tuple(
        (values[later] - values[earlier]) / Decimal(later - earlier)
        for earlier in range(window)
        for later in range(earlier + 1, window)
    )
    return (_median(slopes), rows)


def slope_change(
    series: LogSeries, index: int, *, window: int, decision_time_ns: int
) -> FeatureResult:
    """Current OLS slope minus the OLS slope one window earlier."""
    current = ols_slope(series, index, window=window, decision_time_ns=decision_time_ns)
    previous = ols_slope(series, index - window, window=window, decision_time_ns=decision_time_ns)
    if current is None or previous is None:
        return None
    return (current[0] - previous[0], previous[1] + current[1])


def slope_agreement(*slopes: Decimal | None) -> Decimal | None:
    """Mean sign across scales: one when every slope agrees, zero when they cancel."""
    if not slopes or any(slope is None for slope in slopes):
        return None
    signs = tuple(_sign(slope) for slope in slopes if slope is not None)
    return sum(signs, Decimal(0)) / Decimal(len(signs))


def _admissible(rows: tuple[ResearchBar, ...], decision_time_ns: int) -> bool:
    return all(bar.available_time_ns <= decision_time_ns for bar in rows)


def _median(values: tuple[Decimal, ...]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _sign(value: Decimal) -> Decimal:
    return Decimal(1) if value > 0 else Decimal(-1) if value < 0 else Decimal(0)
```

The unused imports (`math`, `EnrichedBarSample`, `ColumnStatistics`, …) are consumed in Task 4; to keep ruff green in this task, add them in Task 4 instead of now.

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/test_view_features.py -q -p no:cacheprovider
```

Expected: `5 passed`.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/view_features.py tests/test_view_features.py
git commit -m "feat: add causal slope view features"
```

---

### Task 4: View features II — spectral family, `ViewSample`, builder, view matrix

**Files:**
- Modify: `src/trading_bot/view_features.py`
- Modify: `tests/test_view_features.py`

**Interfaces:**
- Produces:

```python
def autocorrelation(series, index, *, window, lag, decision_time_ns) -> FeatureResult
def ljung_box(series, index, *, window, max_lag, decision_time_ns) -> FeatureResult
def funding_phase(series, index) -> tuple[Decimal, Decimal]          # (sin, cos)
def envelope_width(series, index, *, window, confirmation_bars, decision_time_ns) -> FeatureResult

@dataclass(frozen=True, slots=True)
class ViewSample:
    base: EnrichedBarSample
    view_input_source_ids: tuple[str, ...]
    slope_ols_8: Decimal | None
    ...  # one attribute per VIEW_FIELDS entry, same order
    envelope_width_96: Decimal | None
    decision_time_ns -> int (property)

def build_view_samples(enriched: tuple[EnrichedBarSample, ...], primary: tuple[ResearchBar, ...], *, schema: ViewFeatureSchema) -> tuple[ViewSample, ...]
def view_value(sample: ViewSample, field: str) -> Decimal | None

class ViewMatrixSpec: view_schema_hash, fields, preprocessing_formula_id; schema_hash
class FittedViewMatrix: fields, statistics: ColumnStatistics, schema_hash
class TransformedViewMatrix: sample_ids, rows
def fit_view_matrix(samples, *, fields=VIEW_FIELDS, schema=c0_view_feature_schema()) -> FittedViewMatrix
def transform_view_matrix(fitted, samples) -> TransformedViewMatrix
```

- Feature definitions: `acf_lag32_192` = sample autocorrelation of one-bar log returns at lag 32 over the trailing 192 returns; `acf_lag96_384` lag 96 over 384; `acf_lag672_1344` lag 672 over 1,344. `ljung_box_96` = `N(N+2) Σ_{k=1..8} ρ_k² / (N−k)` over 96 returns. `funding_phase_sin/cos` = sine and cosine of `2π · (open_time_ns mod 8h) / 8h` for the decision bar. `envelope_width_96` = `(max high − min low) / close` over the 96 bars that end 8 bars before the decision bar (extrema whose confirmation window has elapsed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_view_features.py`:

```python
from trading_bot.market_features import build_enriched_samples, phase_a_market_feature_schema
from trading_bot.view_features import (
    FUNDING_PERIOD_NS,
    autocorrelation,
    build_view_samples,
    envelope_width,
    fit_view_matrix,
    funding_phase,
    ljung_box,
    transform_view_matrix,
    view_value,
)


def alternating_bars(count: int) -> tuple[ResearchBar, ...]:
    up = Decimal(100) * Decimal("1.002")
    return tuple(bar(index, up if index % 2 else Decimal(100)) for index in range(count))


def test_autocorrelation_of_alternating_returns_is_plus_one_at_even_lags() -> None:
    series = build_log_series(alternating_bars(1_400))
    index = 1_399
    when = decision_time(index)

    even = autocorrelation(series, index, window=192, lag=32, decision_time_ns=when)
    odd = autocorrelation(series, index, window=192, lag=31, decision_time_ns=when)
    weekly = autocorrelation(series, index, window=1_344, lag=672, decision_time_ns=when)

    assert even is not None and odd is not None and weekly is not None
    assert abs(even[0] - Decimal(1)) < Decimal("1e-6")
    assert abs(odd[0] + Decimal(1)) < Decimal("1e-6")
    assert abs(weekly[0] - Decimal(1)) < Decimal("1e-6")
    assert len(even[1]) == 193
    assert autocorrelation(series, 100, window=192, lag=32, decision_time_ns=when) is None


def test_ljung_box_is_positive_for_alternating_returns_and_none_for_flat() -> None:
    series = build_log_series(alternating_bars(200))
    flat = build_log_series(tuple(bar(index, Decimal(100)) for index in range(200)))

    statistic = ljung_box(series, 199, window=96, max_lag=8, decision_time_ns=decision_time(199))

    assert statistic is not None and statistic[0] > Decimal(0)
    assert ljung_box(flat, 199, window=96, max_lag=8, decision_time_ns=decision_time(199)) is None


def test_funding_phase_uses_eight_hour_cycle_of_bar_open_time() -> None:
    bars = (
        bar(0, Decimal(100)),
        ResearchBar("okx-q", "OKX", FUNDING_PERIOD_NS // 4, FUNDING_PERIOD_NS // 4 + 1,
                    Decimal(100), Decimal(101), Decimal(99), Decimal(1)),
    )
    series = build_log_series(bars)

    sin_zero, cos_zero = funding_phase(series, 0)
    sin_quarter, cos_quarter = funding_phase(series, 1)

    assert sin_zero == Decimal(0) and cos_zero == Decimal(1)
    assert abs(sin_quarter - Decimal(1)) < Decimal("1e-12")
    assert abs(cos_quarter) < Decimal("1e-12")


def test_envelope_width_uses_only_confirmed_bars() -> None:
    bars = list(bar(index, Decimal(100)) for index in range(120))
    spike = bars[119]
    bars[119] = ResearchBar(spike.source_id, spike.venue, spike.open_time_ns,
                            spike.available_time_ns, spike.close, Decimal(500), spike.low,
                            spike.base_volume)
    series = build_log_series(tuple(bars))

    width = envelope_width(series, 119, window=96, confirmation_bars=8,
                           decision_time_ns=decision_time(119))

    assert width is not None
    assert width[0] == Decimal("0.02")
    assert width[1][-1].source_id == "okx-111"
    assert envelope_width(series, 100, window=96, confirmation_bars=8,
                          decision_time_ns=decision_time(100)) is None


def test_build_view_samples_aligns_with_enriched_samples_and_reports_inputs() -> None:
    primary = exponential_bars(160)
    enriched = build_enriched_samples(
        primary, (), horizon_bars=4, schema=phase_a_market_feature_schema()
    )
    schema = c0_view_feature_schema()

    views = build_view_samples(enriched, primary, schema=schema)

    assert len(views) == len(enriched)
    assert views[-1].base is enriched[-1]
    assert views[-1].slope_ols_96 is not None
    assert views[-1].acf_lag32_192 is None
    assert views[0].slope_ols_8 is None
    assert "okx-155" in views[-1].view_input_source_ids
    assert view_value(views[-1], "slope_agreement") == Decimal(1)
    with pytest.raises(ValueError, match="unknown view field"):
        view_value(views[-1], "return_4")


def test_view_matrix_fits_on_training_only_and_binds_schema() -> None:
    primary = exponential_bars(400)
    enriched = build_enriched_samples(
        primary, (), horizon_bars=4, schema=phase_a_market_feature_schema()
    )
    views = build_view_samples(enriched, primary, schema=c0_view_feature_schema())

    fitted = fit_view_matrix(views[:300], fields=("slope_ols_8", "acf_lag32_192"))
    late = transform_view_matrix(fitted, views[300:])
    early = transform_view_matrix(fitted, views[:1])

    assert fitted.fields == ("slope_ols_8", "acf_lag32_192")
    assert len(late.rows[0]) == 4
    assert late.rows[0][3] == Decimal(0)
    assert early.rows[0][1] == Decimal(1) and early.rows[0][3] == Decimal(1)
    assert late.sample_ids == tuple(view.base.base.sample_id for view in views[300:])
    assert len(fitted.schema_hash) == 64
    with pytest.raises(ValueError, match="unknown"):
        fit_view_matrix(views[:300], fields=("return_4",))
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_view_features.py -q -p no:cacheprovider
```

Expected: FAIL with `ImportError: cannot import name 'autocorrelation'`.

- [ ] **Step 3: Add the spectral half, the sample type, the builder, and the matrix**

Add the imports promised in Task 3 at the top of `view_features.py` (`math`, `EnrichedBarSample`, `PREPROCESSING_FORMULA_ID`, `ColumnStatistics`, `fit_column_statistics`, `standardize_row`) and append:

```python
def autocorrelation(
    series: LogSeries, index: int, *, window: int, lag: int, decision_time_ns: int
) -> FeatureResult:
    """Sample autocorrelation of one-bar log returns at one predeclared lag."""
    if lag < 1 or lag >= window:
        raise ValueError("autocorrelation lag must be inside the window")
    centered = _centered_returns(series, index, window=window, decision_time_ns=decision_time_ns)
    if centered is None:
        return None
    values, rows = centered
    denominator = sum((value * value for value in values), Decimal(0))
    if denominator == 0:
        return None
    numerator = sum(
        (values[position] * values[position - lag] for position in range(lag, window)),
        Decimal(0),
    )
    return (numerator / denominator, rows)


def ljung_box(
    series: LogSeries, index: int, *, window: int, max_lag: int, decision_time_ns: int
) -> FeatureResult:
    """Ljung-Box portmanteau statistic over the first `max_lag` autocorrelations."""
    if max_lag < 1 or max_lag >= window:
        raise ValueError("ljung-box lag count must be inside the window")
    centered = _centered_returns(series, index, window=window, decision_time_ns=decision_time_ns)
    if centered is None:
        return None
    values, rows = centered
    denominator = sum((value * value for value in values), Decimal(0))
    if denominator == 0:
        return None
    count = Decimal(window)
    statistic = Decimal(0)
    for lag in range(1, max_lag + 1):
        rho = (
            sum(
                (values[position] * values[position - lag] for position in range(lag, window)),
                Decimal(0),
            )
            / denominator
        )
        statistic += rho * rho / (count - Decimal(lag))
    return (count * (count + Decimal(2)) * statistic, rows)


def funding_phase(series: LogSeries, index: int) -> tuple[Decimal, Decimal]:
    """Sine and cosine of the decision bar's position inside the 8-hour funding cycle."""
    fraction = Decimal(series.bars[index].open_time_ns % FUNDING_PERIOD_NS) / Decimal(
        FUNDING_PERIOD_NS
    )
    angle = 2 * math.pi * float(fraction)
    return (Decimal(repr(math.sin(angle))), Decimal(repr(math.cos(angle))))


def envelope_width(
    series: LogSeries,
    index: int,
    *,
    window: int,
    confirmation_bars: int,
    decision_time_ns: int,
) -> FeatureResult:
    """Channel width over bars whose extremum status could have been confirmed by now."""
    end = index - confirmation_bars
    start = end - window + 1
    if start < 0:
        return None
    rows = series.bars[start : end + 1]
    current = series.bars[index]
    if not _admissible((*rows, current), decision_time_ns):
        return None
    highest = max(bar.high for bar in rows)
    lowest = min(bar.low for bar in rows)
    return ((highest - lowest) / current.close, rows)


def _centered_returns(
    series: LogSeries, index: int, *, window: int, decision_time_ns: int
) -> tuple[tuple[Decimal, ...], tuple[ResearchBar, ...]] | None:
    start = index - window + 1
    if start < 1:
        return None
    rows = series.bars[start - 1 : index + 1]
    if not _admissible(rows, decision_time_ns):
        return None
    returns = series.log_return[start : index + 1]
    mean = sum(returns, Decimal(0)) / Decimal(window)
    return (tuple(value - mean for value in returns), rows)


@dataclass(frozen=True, slots=True)
class ViewSample:
    """Decision-time-admissible slope and spectral views attached to an enriched sample."""

    base: EnrichedBarSample
    view_input_source_ids: tuple[str, ...]
    slope_ols_8: Decimal | None
    slope_ols_32: Decimal | None
    slope_ols_96: Decimal | None
    slope_theil_sen_8: Decimal | None
    slope_theil_sen_32: Decimal | None
    slope_change_8: Decimal | None
    slope_agreement: Decimal | None
    acf_lag32_192: Decimal | None
    acf_lag96_384: Decimal | None
    acf_lag672_1344: Decimal | None
    funding_phase_sin: Decimal | None
    funding_phase_cos: Decimal | None
    ljung_box_96: Decimal | None
    envelope_width_96: Decimal | None

    @property
    def decision_time_ns(self) -> int:
        return self.base.decision_time_ns


def build_view_samples(
    enriched: tuple[EnrichedBarSample, ...],
    primary: tuple[ResearchBar, ...],
    *,
    schema: ViewFeatureSchema,
) -> tuple[ViewSample, ...]:
    """Attach view features using only bars available at each sample's decision time."""
    if schema != c0_view_feature_schema():
        raise ValueError("view samples require the canonical C.0 view schema")
    series = build_log_series(primary)
    views: list[ViewSample] = []
    for sample in enriched:
        if len(sample.base.input_source_ids) < 2:
            raise ValueError("enriched sample does not reference a primary bar")
        index = series.index_by_source_id.get(sample.base.input_source_ids[1])
        if index is None:
            raise ValueError("enriched sample references an unknown primary bar")
        when = sample.decision_time_ns
        rows: list[ResearchBar] = []
        ols_8 = ols_slope(series, index, window=8, decision_time_ns=when)
        ols_32 = ols_slope(series, index, window=32, decision_time_ns=when)
        ols_96 = ols_slope(series, index, window=96, decision_time_ns=when)
        theil_8 = theil_sen_slope(series, index, window=8, decision_time_ns=when)
        theil_32 = theil_sen_slope(series, index, window=32, decision_time_ns=when)
        change_8 = slope_change(series, index, window=8, decision_time_ns=when)
        acf_32 = autocorrelation(series, index, window=192, lag=32, decision_time_ns=when)
        acf_96 = autocorrelation(series, index, window=384, lag=96, decision_time_ns=when)
        acf_672 = autocorrelation(series, index, window=1_344, lag=672, decision_time_ns=when)
        box = ljung_box(series, index, window=96, max_lag=8, decision_time_ns=when)
        envelope = envelope_width(
            series,
            index,
            window=96,
            confirmation_bars=EXTREMUM_CONFIRMATION_BARS,
            decision_time_ns=when,
        )
        phase_sin, phase_cos = funding_phase(series, index)
        for result in (
            ols_8, ols_32, ols_96, theil_8, theil_32, change_8, acf_32, acf_96, acf_672, box,
            envelope,
        ):
            _include_rows(rows, result)
        views.append(
            ViewSample(
                base=sample,
                view_input_source_ids=tuple(bar.source_id for bar in rows),
                slope_ols_8=_value(ols_8),
                slope_ols_32=_value(ols_32),
                slope_ols_96=_value(ols_96),
                slope_theil_sen_8=_value(theil_8),
                slope_theil_sen_32=_value(theil_32),
                slope_change_8=_value(change_8),
                slope_agreement=slope_agreement(_value(ols_8), _value(ols_32), _value(ols_96)),
                acf_lag32_192=_value(acf_32),
                acf_lag96_384=_value(acf_96),
                acf_lag672_1344=_value(acf_672),
                funding_phase_sin=phase_sin,
                funding_phase_cos=phase_cos,
                ljung_box_96=_value(box),
                envelope_width_96=_value(envelope),
            )
        )
    return tuple(views)


def view_value(sample: ViewSample, field: str) -> Decimal | None:
    """Read one predeclared view field, failing closed on unknown or non-finite values."""
    if field not in VIEW_FIELDS:
        raise ValueError(f"unknown view field: {field!r}")
    value: object = getattr(sample, field)
    if value is None:
        return None
    if not isinstance(value, Decimal):
        raise TypeError(f"view field {field!r} must be Decimal or None")
    if not value.is_finite():
        raise ValueError(f"view field {field!r} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ViewMatrixSpec:
    """Canonical identity of an ordered view-vector preprocessing contract."""

    view_schema_hash: str
    fields: tuple[str, ...]
    preprocessing_formula_id: str = PREPROCESSING_FORMULA_ID

    @property
    def schema_hash(self) -> str:
        return content_sha256(
            {
                "view_schema_hash": self.view_schema_hash,
                "fields": list(self.fields),
                "preprocessing_formula_id": self.preprocessing_formula_id,
            }
        )


@dataclass(frozen=True, slots=True)
class FittedViewMatrix:
    fields: tuple[str, ...]
    statistics: ColumnStatistics
    schema_hash: str


@dataclass(frozen=True, slots=True)
class TransformedViewMatrix:
    sample_ids: tuple[str, ...]
    rows: tuple[tuple[Decimal, ...], ...]


def fit_view_matrix(
    samples: tuple[ViewSample, ...],
    *,
    fields: tuple[str, ...] = VIEW_FIELDS,
    schema: ViewFeatureSchema | None = None,
) -> FittedViewMatrix:
    """Fit view imputation and standardization from training samples only."""
    resolved = c0_view_feature_schema() if schema is None else schema
    if not samples:
        raise ValueError("view-matrix training samples must not be empty")
    _validate_view_fields(fields)
    _validate_unique_view_ids(samples)
    columns = tuple(tuple(view_value(sample, field) for sample in samples) for field in fields)
    statistics = fit_column_statistics(columns, labels=fields)
    return FittedViewMatrix(
        fields=fields,
        statistics=statistics,
        schema_hash=ViewMatrixSpec(resolved.schema_hash, fields).schema_hash,
    )


def transform_view_matrix(
    fitted: FittedViewMatrix, samples: tuple[ViewSample, ...]
) -> TransformedViewMatrix:
    """Apply frozen training statistics to any sample set."""
    _validate_view_fields(fitted.fields)
    _validate_unique_view_ids(samples)
    rows = tuple(
        standardize_row(
            fitted.statistics, tuple(view_value(sample, field) for field in fitted.fields)
        )
        for sample in samples
    )
    return TransformedViewMatrix(
        sample_ids=tuple(sample.base.base.sample_id for sample in samples), rows=rows
    )


def _validate_view_fields(fields: tuple[str, ...]) -> None:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("view fields must be non-empty and unique")
    unknown = tuple(field for field in fields if field not in VIEW_FIELDS)
    if unknown:
        raise ValueError(f"unknown view fields: {unknown!r}")


def _validate_unique_view_ids(samples: tuple[ViewSample, ...]) -> None:
    sample_ids = tuple(sample.base.base.sample_id for sample in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("view-matrix sample IDs must be unique")


def _include_rows(rows: list[ResearchBar], result: FeatureResult) -> None:
    if result is None:
        return
    known = {bar.source_id for bar in rows}
    rows.extend(bar for bar in result[1] if bar.source_id not in known)


def _value(result: FeatureResult) -> Decimal | None:
    return None if result is None else result[0]
```

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/test_view_features.py -q -p no:cacheprovider
```

Expected: `11 passed` (the 1,400-bar autocorrelation test takes a few seconds; that is the Decimal cost of the weekly lag).

- [ ] **Step 5: Correct the spec table to match the implemented contract**

In `docs/superpowers/specs/2026-09-08-ctm-moe-challenger-design.md` section 6.2, replace the E1 row's feature text with `OLS slope over 8/32/96 bars; Theil-Sen slope over 8/32 bars; slope-of-slope over 8 bars; scale agreement (mean sign across the three OLS windows)` and the E3/E4 learner cells with `zero-init residual head (Task 9)`; in section 6.5 replace the expected-return mapping sentence with `p is mapped to an expected return as p·mean_positive_train + (1−p)·mean_nonpositive_train, the same conditional-mean mapping the Phase-A logistic already uses.`

- [ ] **Step 6: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/view_features.py tests/test_view_features.py docs/superpowers/specs/2026-09-08-ctm-moe-challenger-design.md
git commit -m "feat: add spectral view features and view matrix"
```

---

### Task 5: Registry — neural family name and `neural_fold_runs` ledger

**Files:**
- Modify: `src/trading_bot/hypothesis_registry.py`
- Modify: `src/trading_bot/phase_a_fold_run.py:361-470` (`_verify_inputs_and_attempt`)
- Modify: `tests/test_hypothesis_registry.py`
- Modify: `tests/test_phase_a_fold_run.py`

**Interfaces:**
- Produces: `HypothesisFamily.name` accepts `"ctm_regime_gated_mixture"`;

```python
class NeuralFoldRun(BaseModel):
    attempt_id: UUID
    horizon_bars: Literal[4, 16]
    encoder_kind: Literal["none", "gru", "ctm"]
    gate_kind: Literal["none", "uniform", "best_single", "learned"]
    seed_ordinal: Literal[0, 1, 2]
    fold_index: Literal[0, 1, 2]
    status: Literal["completed", "failed"]
    artifact_hash: Sha256Hex | None
    error_code: str | None
    recorded_at_ns: int

HypothesisRegistry.record_neural_fold_run(run: NeuralFoldRun) -> NeuralFoldRun
HypothesisRegistry.list_neural_fold_runs() -> tuple[NeuralFoldRun, ...]
```

- `run_phase_a_fold` fails closed with `error_code == "FAMILY_NOT_PHASE_A"` when the reserved family is not one of the three Phase-A families.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hypothesis_registry.py` (reuse that module's existing family fixture helper; if it is named differently, adapt the call):

```python
from trading_bot.hypothesis_registry import NeuralFoldRun
from trading_bot.registry import RegistryConflictError


def _neural_run(attempt_id: UUID, **overrides: object) -> NeuralFoldRun:
    values: dict[str, object] = {
        "attempt_id": attempt_id,
        "horizon_bars": 4,
        "encoder_kind": "gru",
        "gate_kind": "learned",
        "seed_ordinal": 0,
        "fold_index": 0,
        "status": "completed",
        "artifact_hash": "e" * 64,
        "error_code": None,
        "recorded_at_ns": 5,
    }
    values.update(overrides)
    return NeuralFoldRun.model_validate(values)


def test_neural_family_name_is_accepted() -> None:
    family = HypothesisFamily(
        family_id=UUID("00000000-0000-0000-0000-000000000901"),
        name="ctm_regime_gated_mixture",
        statement="regime-gated residual mixture",
        feature_schema_hash="a" * 64,
        audit_report_hash="b" * 64,
        h4_split_manifest_hash="c" * 64,
        h16_split_manifest_hash="d" * 64,
    )

    assert family.name == "ctm_regime_gated_mixture"


def test_neural_fold_runs_are_keyed_by_encoder_gate_seed_and_fold(tmp_path: Path) -> None:
    family = HypothesisFamily(
        family_id=UUID("00000000-0000-0000-0000-000000000901"),
        name="ctm_regime_gated_mixture",
        statement="regime-gated residual mixture",
        feature_schema_hash="a" * 64,
        audit_report_hash="b" * 64,
        h4_split_manifest_hash="c" * 64,
        h16_split_manifest_hash="d" * 64,
    )
    with HypothesisRegistry(tmp_path / "neural.sqlite3") as registry:
        attempt = registry.reserve_attempt(family, config_hash="f" * 64, revision_reason=None)
        first = registry.record_neural_fold_run(_neural_run(attempt.attempt_id))
        replay = registry.record_neural_fold_run(_neural_run(attempt.attempt_id))
        other_seed = registry.record_neural_fold_run(
            _neural_run(attempt.attempt_id, seed_ordinal=1, artifact_hash="1" * 64)
        )
        with pytest.raises(RegistryConflictError, match="recorded differently"):
            registry.record_neural_fold_run(
                _neural_run(attempt.attempt_id, artifact_hash="2" * 64)
            )
        runs = registry.list_neural_fold_runs()

    assert first == replay
    assert runs == (first, other_seed)


def test_neural_fold_run_requires_reserved_attempt_and_valid_outcome(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only an artifact hash"):
        _neural_run(UUID(int=1), error_code="X")
    with pytest.raises(ValueError, match="bounded nonempty error code"):
        _neural_run(UUID(int=1), status="failed", artifact_hash=None, error_code=" ")
    with HypothesisRegistry(tmp_path / "neural.sqlite3") as registry:
        with pytest.raises(RegistryConflictError, match="must be reserved"):
            registry.record_neural_fold_run(_neural_run(UUID(int=1)))
```

Append to `tests/test_phase_a_fold_run.py`:

```python
def test_phase_a_fold_rejects_non_phase_a_family(tmp_path: Path) -> None:
    config = phase_a_fold_fixture(tmp_path)
    with HypothesisRegistry(config.registry_path) as registry:
        stored = registry.get_family(config.hypothesis_family_id)
    assert stored is not None
    neural = HypothesisFamily(
        family_id=UUID("00000000-0000-0000-0000-000000000901"),
        name="ctm_regime_gated_mixture",
        statement="regime-gated residual mixture",
        feature_schema_hash=stored.feature_schema_hash,
        audit_report_hash=stored.audit_report_hash,
        h4_split_manifest_hash=stored.h4_split_manifest_hash,
        h16_split_manifest_hash=stored.h16_split_manifest_hash,
    )
    with HypothesisRegistry(config.registry_path) as registry:
        attempt = registry.reserve_attempt(neural, config_hash="9" * 64, revision_reason=None)

    with pytest.raises(PhaseAFoldError) as raised:
        run_phase_a_fold(
            replace(config, hypothesis_family_id=neural.family_id, attempt_id=attempt.attempt_id)
        )

    assert raised.value.error_code == "FAMILY_NOT_PHASE_A"
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_hypothesis_registry.py tests/test_phase_a_fold_run.py -q -p no:cacheprovider -k "neural or non_phase_a"
```

Expected: FAIL (`ImportError: NeuralFoldRun` and a pydantic literal error on the family name).

- [ ] **Step 3: Widen the family name and add the neural ledger**

In `hypothesis_registry.py`:

1. `HypothesisFamily.name` literal gains `"ctm_regime_gated_mixture"`.
2. Extract the outcome validation into a module function and reuse it:

```python
def _validate_run_outcome(
    status: str, artifact_hash: str | None, error_code: str | None
) -> None:
    if status == "completed" and (artifact_hash is None or error_code is not None):
        raise ValueError("completed fold run requires only an artifact hash")
    if status == "failed" and (
        artifact_hash is not None
        or error_code is None
        or not error_code.strip()
        or len(error_code) > _MAX_ERROR_CODE_LENGTH
    ):
        raise ValueError("failed fold run requires a bounded nonempty error code only")
```

   `HypothesisFoldRun.validate_outcome` becomes `_validate_run_outcome(self.status, self.artifact_hash, self.error_code); return self`.

3. Add after `HypothesisFoldRun`:

```python
class NeuralFoldRun(BaseModel):
    """An immutable outcome for one neural campaign cell (encoder, gate, seed, fold)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: UUID
    horizon_bars: Literal[4, 16]
    encoder_kind: Literal["none", "gru", "ctm"]
    gate_kind: Literal["none", "uniform", "best_single", "learned"]
    seed_ordinal: Literal[0, 1, 2]
    fold_index: Literal[0, 1, 2]
    status: Literal["completed", "failed"]
    artifact_hash: Sha256Hex | None
    error_code: str | None
    recorded_at_ns: int

    @field_validator("recorded_at_ns")
    @classmethod
    def validate_recorded_at_ns(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("fold run record time must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        _validate_run_outcome(self.status, self.artifact_hash, self.error_code)
        return self
```

4. In `_create_tables`, add a fourth statement:

```python
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_fold_runs (
                    attempt_id TEXT NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    encoder_kind TEXT NOT NULL,
                    gate_kind TEXT NOT NULL,
                    seed_ordinal INTEGER NOT NULL,
                    fold_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    artifact_hash TEXT,
                    error_code TEXT,
                    recorded_at_ns INTEGER NOT NULL,
                    FOREIGN KEY (attempt_id) REFERENCES hypothesis_attempts(attempt_id),
                    UNIQUE (attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal,
                            fold_index),
                    CHECK (horizon_bars IN (4, 16)),
                    CHECK (encoder_kind IN ('none', 'gru', 'ctm')),
                    CHECK (gate_kind IN ('none', 'uniform', 'best_single', 'learned')),
                    CHECK (seed_ordinal IN (0, 1, 2)),
                    CHECK (fold_index IN (0, 1, 2)),
                    CHECK (
                        (status = 'completed' AND artifact_hash IS NOT NULL AND error_code IS NULL)
                        OR (
                            status = 'failed'
                            AND artifact_hash IS NULL
                            AND length(trim(error_code)) > 0
                            AND length(error_code) <= 128
                        )
                    )
                )
                """
            )
```

5. Add the methods (mirroring `record_fold_run` / `list_fold_runs` / `_get_fold_run` exactly, with the six-column key and `ORDER BY attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal, fold_index`):

```python
    def list_neural_fold_runs(self) -> tuple[NeuralFoldRun, ...]:
        rows = self._require_connection().execute(
            """
            SELECT attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal, fold_index,
                   status, artifact_hash, error_code, recorded_at_ns
            FROM neural_fold_runs
            ORDER BY attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal, fold_index
            """
        ).fetchall()
        return tuple(_neural_run_from_row(row) for row in rows)

    def record_neural_fold_run(self, fold_run: NeuralFoldRun) -> NeuralFoldRun:
        """Append one immutable neural cell outcome, allowing exact replay only."""
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            attempt_row = connection.execute(
                "SELECT 1 FROM hypothesis_attempts WHERE attempt_id = ?",
                (str(fold_run.attempt_id),),
            ).fetchone()
            if attempt_row is None:
                raise RegistryConflictError(
                    f"hypothesis attempt {fold_run.attempt_id} must be reserved before "
                    "recording folds"
                )
            existing = self._get_neural_fold_run(fold_run)
            if existing is not None:
                if existing != fold_run:
                    raise RegistryConflictError(
                        "neural fold run is already recorded differently"
                    )
                connection.rollback()
                return existing
            connection.execute(
                """
                INSERT INTO neural_fold_runs (
                    attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal, fold_index,
                    status, artifact_hash, error_code, recorded_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(fold_run.attempt_id),
                    fold_run.horizon_bars,
                    fold_run.encoder_kind,
                    fold_run.gate_kind,
                    fold_run.seed_ordinal,
                    fold_run.fold_index,
                    fold_run.status,
                    fold_run.artifact_hash,
                    fold_run.error_code,
                    fold_run.recorded_at_ns,
                ),
            )
        except BaseException:
            connection.rollback()
            raise
        connection.commit()
        return fold_run

    def _get_neural_fold_run(self, fold_run: NeuralFoldRun) -> NeuralFoldRun | None:
        row = self._require_connection().execute(
            """
            SELECT attempt_id, horizon_bars, encoder_kind, gate_kind, seed_ordinal, fold_index,
                   status, artifact_hash, error_code, recorded_at_ns
            FROM neural_fold_runs
            WHERE attempt_id = ? AND horizon_bars = ? AND encoder_kind = ? AND gate_kind = ?
              AND seed_ordinal = ? AND fold_index = ?
            """,
            (
                str(fold_run.attempt_id),
                fold_run.horizon_bars,
                fold_run.encoder_kind,
                fold_run.gate_kind,
                fold_run.seed_ordinal,
                fold_run.fold_index,
            ),
        ).fetchone()
        return None if row is None else _neural_run_from_row(row)
```

and the module-level row mapper:

```python
def _neural_run_from_row(row: tuple[object, ...]) -> NeuralFoldRun:
    return NeuralFoldRun.model_validate(
        {
            "attempt_id": row[0],
            "horizon_bars": row[1],
            "encoder_kind": row[2],
            "gate_kind": row[3],
            "seed_ordinal": row[4],
            "fold_index": row[5],
            "status": row[6],
            "artifact_hash": row[7],
            "error_code": row[8],
            "recorded_at_ns": row[9],
        }
    )
```

In `phase_a_fold_run._verify_inputs_and_attempt`, directly after the `family is None` check:

```python
    if family.name not in _FAMILY_FIELDS:
        _fail("hypothesis family is not a phase-a family", "FAMILY_NOT_PHASE_A")
```

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/test_hypothesis_registry.py tests/test_phase_a_fold_run.py tests/test_phase_a_pipeline.py tests/test_phase_a_decision.py -q -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/hypothesis_registry.py src/trading_bot/phase_a_fold_run.py tests/test_hypothesis_registry.py tests/test_phase_a_fold_run.py
git commit -m "feat: add neural fold-run ledger to hypothesis registry"
```

---

### Task 6: Extract `verify_research_inputs` from the Phase-A fold runner

**Files:**
- Modify: `src/trading_bot/phase_a_fold_run.py:196-224` (`_AuditEvidence`, `_VerifiedInputs`) and `:361-465` (`_verify_inputs_and_attempt`)
- Modify: `tests/test_phase_a_fold_run.py`

**Interfaces:**
- Produces (public, consumed by `neural/fold_run.py`):

```python
@dataclass(frozen=True, slots=True)
class AuditEvidence:            # renamed from _AuditEvidence, same fields
    report_hash: str
    capture_root_hash: str
    dataset_root_hash: str
    cost_capture_root_hash: str
    base_costs: CostScenario
    adverse_costs: CostScenario
    base_spread_bps: Decimal
    adverse_spread_bps: Decimal
    audited_costs: dict[str, object]

@dataclass(frozen=True, slots=True)
class ResearchInputs:
    capture_hash: str
    dataset_hash: str
    capture_manifest: dict[str, object]
    dataset_manifest: dict[str, object]
    split: dict[str, object]
    split_hash: str
    fold: dict[str, object]
    horizon_bars: Literal[4, 16]
    schema: MarketFeatureSchema
    audit: AuditEvidence

def verify_research_inputs(
    *, capture_root: Path, split_manifest_path: Path, audit_path: Path, fold_index: int
) -> ResearchInputs: ...
```

  Raises `PhaseAFoldError` with the existing codes (`CAPTURE_VERIFICATION_FAILED`, `SPLIT_VERIFICATION_FAILED`, `AUDIT_VERIFICATION_FAILED`, `SPLIT_CAPTURE_LINK_MISMATCH`, `AUDIT_CAPTURE_LINK_MISMATCH`, `AUDIT_INELIGIBLE`, `HORIZON_INVALID`).

- The neural runner also imports these private helpers unchanged: `_split_validation`, `_select_samples`, `_validate_disjoint_manifest_memberships`, `_membership_hash`, `_decisions`, `_select_floor`, `_signals_and_reasons_at_floor`, `_bootstrap`, `_reason_codes`, `_write_immutable`, `_economic_evidence_record`, `_interval_record`, `_family_record`, `_attempt_record`, `_market_schema_record`, `_int_field`, `_list_field`, `_string_list_field`. Do not rename them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase_a_fold_run.py`:

```python
from trading_bot.phase_a_fold_run import verify_research_inputs


def test_verify_research_inputs_returns_fold_and_audit_evidence(tmp_path: Path) -> None:
    config = phase_a_fold_fixture(tmp_path)

    inputs = verify_research_inputs(
        capture_root=config.capture_root,
        split_manifest_path=config.split_manifest_path,
        audit_path=config.audit_path,
        fold_index=0,
    )

    assert inputs.horizon_bars == 4
    assert inputs.fold["fold_index"] == 0
    assert inputs.audit.base_costs.name == "base"
    assert inputs.split["manifest_hash"] == inputs.split_hash
    with pytest.raises(PhaseAFoldError) as raised:
        verify_research_inputs(
            capture_root=config.capture_root,
            split_manifest_path=config.split_manifest_path,
            audit_path=tmp_path / "missing.json",
            fold_index=0,
        )
    assert raised.value.error_code == "AUDIT_VERIFICATION_FAILED"
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_phase_a_fold_run.py -q -p no:cacheprovider -k verify_research_inputs
```

Expected: FAIL with `ImportError: cannot import name 'verify_research_inputs'`.

- [ ] **Step 3: Extract the function**

1. Rename `_AuditEvidence` to `AuditEvidence` everywhere in `phase_a_fold_run.py` (`grep -n _AuditEvidence src tests` must return nothing afterwards; update tests if they reference it).
2. Add `ResearchInputs` (fields above) directly before `_VerifiedInputs`.
3. Move the first half of `_verify_inputs_and_attempt` (from `verify_candle_capture` through `schema = phase_a_market_feature_schema()`) into:

```python
def verify_research_inputs(
    *, capture_root: Path, split_manifest_path: Path, audit_path: Path, fold_index: int
) -> ResearchInputs:
    """Verify capture, split, and audit linkage for one fold without touching a registry."""
    if not verify_candle_capture(capture_root).valid:
        _fail("phase-a capture verification failed", "CAPTURE_VERIFICATION_FAILED")
    if not verify_walk_forward_manifest(split_manifest_path):
        _fail("phase-a split verification failed", "SPLIT_VERIFICATION_FAILED")
    if not verify_phase_a_audit(audit_path):
        _fail("phase-a audit verification failed", "AUDIT_VERIFICATION_FAILED")
    capture = _load_object(capture_root / "capture-manifest.json", "capture manifest")
    dataset = _load_object(capture_root / "dataset" / "dataset-manifest.json", "dataset manifest")
    split = _load_object(split_manifest_path, "split manifest")
    audit_document = _load_object(audit_path, "phase-a audit")
    capture_hash = _sha256_field(capture, "capture_root_hash")
    dataset_hash = _sha256_field(dataset, "root_hash")
    split_hash = _sha256_field(split, "manifest_hash")
    if split.get("capture_root_hash") != capture_hash or split.get("dataset_root_hash") != dataset_hash:
        _fail("phase-a split linkage does not match capture", "SPLIT_CAPTURE_LINK_MISMATCH")
    if (
        audit_document.get("capture_root_hash") != capture_hash
        or audit_document.get("dataset_root_hash") != dataset_hash
    ):
        _fail("phase-a audit linkage does not match capture", "AUDIT_CAPTURE_LINK_MISMATCH")
    if audit_document.get("eligibility") != "eligible":
        _fail("phase-a audit is ineligible", "AUDIT_INELIGIBLE")
    audit = _audit_evidence(audit_document)
    try:
        raw_horizon = manifest_horizon_bars(split)
    except ValueError as error:
        raise PhaseAFoldError(str(error), error_code="HORIZON_INVALID") from error
    if raw_horizon not in (4, 16):
        _fail("phase-a horizon must be h4 or h16", "HORIZON_INVALID")
    horizon_bars: Literal[4, 16] = 4 if raw_horizon == 4 else 16
    return ResearchInputs(
        capture_hash=capture_hash,
        dataset_hash=dataset_hash,
        capture_manifest=capture,
        dataset_manifest=dataset,
        split=split,
        split_hash=split_hash,
        fold=_select_fold(_list_field(split, "folds"), fold_index),
        horizon_bars=horizon_bars,
        schema=phase_a_market_feature_schema(),
        audit=audit,
    )
```

4. `_verify_inputs_and_attempt` starts with `research = verify_research_inputs(capture_root=config.capture_root, split_manifest_path=config.split_manifest_path, audit_path=config.audit_path, fold_index=config.fold_index)`, keeps the registry block unchanged reading `research.audit`, `research.schema`, `research.split_hash`, `research.horizon_bars`, and returns `_VerifiedInputs(research.capture_hash, research.dataset_hash, research.capture_manifest, research.dataset_manifest, research.split, research.split_hash, research.fold, research.horizon_bars, family, attempt, research.schema, research.audit)`.

- [ ] **Step 4: Run the full suite**

```powershell
uv run pytest -q -p no:cacheprovider
```

Expected: everything passes (the Phase-A runner tests are the behavioral guard for this refactor).

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/phase_a_fold_run.py tests/test_phase_a_fold_run.py
git commit -m "refactor: extract research input verification"
```

### Task 7: Frozen mixture spec, family, campaign hash, registration (torch-free)

**Files:**
- Create: `src/trading_bot/neural/spec.py`
- Create: `tests/test_neural_spec.py` (torch-free, outside `tests/neural`)

**Interfaces:**
- Produces:

```python
type ExpertKind = Literal["slope", "spectral", "rate", "summary"]
type EncoderKind = Literal["none", "gru", "ctm"]
type GateKind = Literal["none", "uniform", "best_single", "learned"]
EXPERT_ORDER: tuple[ExpertKind, ...] = ("slope", "spectral", "rate", "summary")
NEURAL_FAMILY_NAME = "ctm_regime_gated_mixture"
VRAM_LIMIT_BYTES = 12_884_901_888

class CtmMixtureSpec(BaseModel):   # frozen, extra="forbid"; fields listed in Step 3
    def canonical_record(self) -> dict[str, object]
    spec_hash: str (property)

def neural_family(*, family_id: UUID, feature_schema_hash: str, audit_report_hash: str,
                  h4_split_manifest_hash: str, h16_split_manifest_hash: str) -> HypothesisFamily
def neural_campaign_config_hash(family: HypothesisFamily, spec: CtmMixtureSpec, *,
                                block_length: int, bootstrap_repetitions: int,
                                minimum_selection_trades: int) -> str
def register_neural_campaign(registry_path: Path, family: HypothesisFamily, spec: CtmMixtureSpec, *,
                             block_length: int = 96, bootstrap_repetitions: int = 2000,
                             minimum_selection_trades: int = 200,
                             revision_reason: str | None = None) -> HypothesisAttempt
def load_mixture_spec(path: Path) -> CtmMixtureSpec
```

- Deviation recorded here: `reference_source_hash` is `Sha256Hex | None` (None until Plan 2 vendors the CTM source); `max_epochs` is a bounded `int` (1–30) instead of `Literal[30]` so tests can shorten training; the frozen campaign file pins 30.

- [ ] **Step 1: Write the failing tests**

`tests/test_neural_spec.py`:

```python
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from trading_bot.hypothesis_registry import HypothesisBudgetError, HypothesisRegistry
from trading_bot.neural.spec import (
    EXPERT_ORDER,
    CtmMixtureSpec,
    load_mixture_spec,
    neural_campaign_config_hash,
    neural_family,
    register_neural_campaign,
)

FAMILY_ID = UUID("00000000-0000-0000-0000-000000000901")


def spec(**overrides: object) -> CtmMixtureSpec:
    values: dict[str, object] = {
        "version": "1.0.0",
        "reference_commit": "0" * 40,
        "feature_schema_hash": "a" * 64,
        "view_schema_hash": "b" * 64,
        "certainty_threshold_grid": (Decimal("0"), Decimal("0.05"), Decimal("0.10")),
        "seeds": (11, 12, 13),
    }
    values.update(overrides)
    return CtmMixtureSpec.model_validate(values)


def family():
    return neural_family(
        family_id=FAMILY_ID,
        feature_schema_hash="a" * 64,
        audit_report_hash="c" * 64,
        h4_split_manifest_hash="d" * 64,
        h16_split_manifest_hash="e" * 64,
    )


def test_spec_defaults_match_frozen_design() -> None:
    frozen = spec()

    assert frozen.experts == EXPERT_ORDER
    assert frozen.window_bars == 96 and frozen.d_model == 256 and frozen.iterations == 20
    assert frozen.batch_size == 256 and frozen.patience == 5 and frozen.max_epochs == 30
    assert frozen.load_balance_weight == Decimal("0.01")
    assert frozen.reference_source_hash is None
    assert len(frozen.spec_hash) == 64


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"experts": ("summary", "slope", "spectral", "rate")}, "frozen expert order"),
        ({"seeds": (1, 1, 2)}, "distinct"),
        ({"certainty_threshold_grid": (Decimal("0.2"), Decimal("0.1"))}, "strictly increasing"),
        ({"certainty_threshold_grid": (Decimal("1"),)}, "below one"),
        ({"load_balance_weight": Decimal("1")}, "load balance"),
        ({"learning_rate": Decimal("0.5")}, "learning rate"),
        ({"max_epochs": 31}, "epochs"),
        ({"reference_commit": "xyz"}, "commit"),
        ({"window_bars": 64}, "window_bars"),
    ),
)
def test_spec_rejects_unfrozen_values(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        spec(**overrides)


def test_campaign_hash_binds_family_spec_and_gates() -> None:
    base = neural_campaign_config_hash(
        family(), spec(), block_length=96, bootstrap_repetitions=2000, minimum_selection_trades=200
    )
    changed_spec = neural_campaign_config_hash(
        family(), spec(load_balance_weight=Decimal("0.02")),
        block_length=96, bootstrap_repetitions=2000, minimum_selection_trades=200,
    )
    changed_gate = neural_campaign_config_hash(
        family(), spec(), block_length=96, bootstrap_repetitions=2000, minimum_selection_trades=199
    )

    assert len({base, changed_spec, changed_gate}) == 3


def test_register_neural_campaign_respects_two_attempt_budget(tmp_path: Path) -> None:
    registry_path = tmp_path / "neural.sqlite3"

    primary = register_neural_campaign(registry_path, family(), spec())
    revision = register_neural_campaign(
        registry_path, family(), spec(window_bars=96), revision_reason="inner blocked CV"
    )
    with pytest.raises(HypothesisBudgetError):
        register_neural_campaign(registry_path, family(), spec(), revision_reason="third")
    with HypothesisRegistry(registry_path) as registry:
        stored = registry.get_family(FAMILY_ID)

    assert primary.attempt_index == 0 and revision.attempt_index == 1
    assert stored is not None and stored.name == "ctm_regime_gated_mixture"
    assert primary.config_hash == neural_campaign_config_hash(
        family(), spec(), block_length=96, bootstrap_repetitions=2000, minimum_selection_trades=200
    )


def test_load_mixture_spec_round_trips_and_rejects_json_floats(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec().canonical_record()), encoding="utf-8")

    assert load_mixture_spec(path) == spec()

    path.write_text(
        json.dumps({**spec().canonical_record(), "learning_rate": 0.001}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="strings or integers"):
        load_mixture_spec(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_neural_spec.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.neural.spec'`.

- [ ] **Step 3: Write `neural/spec.py`**

```python
"""Frozen C.0 mixture specification, campaign identity, and registration (torch-free)."""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from trading_bot.canonical import content_sha256
from trading_bot.hypothesis_registry import HypothesisAttempt, HypothesisFamily, HypothesisRegistry
from trading_bot.registry import Sha256Hex

type ExpertKind = Literal["slope", "spectral", "rate", "summary"]
type EncoderKind = Literal["none", "gru", "ctm"]
type GateKind = Literal["none", "uniform", "best_single", "learned"]

EXPERT_ORDER: tuple[ExpertKind, ...] = ("slope", "spectral", "rate", "summary")
NEURAL_FAMILY_NAME = "ctm_regime_gated_mixture"
NEURAL_FAMILY_STATEMENT = (
    "a regime-gated residual mixture with a sequence encoder adds adverse-cost OOS value"
)
VRAM_LIMIT_BYTES = 12_884_901_888
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class CtmMixtureSpec(BaseModel):
    """One immutable mixture configuration shared by every cell of a campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["1.0.0"]
    reference_commit: str
    reference_source_hash: Sha256Hex | None = None
    feature_schema_hash: Sha256Hex
    view_schema_hash: Sha256Hex
    window_bars: Literal[96] = 96
    d_model: Literal[256] = 256
    d_input: Literal[64] = 64
    heads: Literal[4] = 4
    iterations: Literal[20] = 20
    memory_length: Literal[16] = 16
    n_synch_out: Literal[64] = 64
    n_synch_action: Literal[32] = 32
    gru_hidden: Literal[64] = 64
    expert_hidden: Literal[8] = 8
    experts: tuple[ExpertKind, ...] = EXPERT_ORDER
    load_balance_weight: Decimal = Decimal("0.01")
    certainty_threshold_grid: tuple[Decimal, ...]
    seeds: tuple[int, int, int]
    batch_size: Literal[256] = 256
    learning_rate: Decimal = Decimal("0.001")
    max_epochs: int = 30
    patience: Literal[5] = 5
    vram_limit_bytes: Literal[12884901888] = VRAM_LIMIT_BYTES

    @field_validator("reference_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not _COMMIT_PATTERN.match(value):
            raise ValueError("reference commit must be a 40-character lowercase hex git SHA")
        return value

    @model_validator(mode="after")
    def validate_frozen_shape(self) -> Self:
        if self.experts != EXPERT_ORDER:
            raise ValueError("experts must use the frozen expert order")
        if len(set(self.seeds)) != 3 or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be three distinct non-negative integers")
        grid = self.certainty_threshold_grid
        if not grid or any(not value.is_finite() for value in grid):
            raise ValueError("certainty threshold grid must be non-empty and finite")
        if any(later <= earlier for earlier, later in zip(grid, grid[1:], strict=False)):
            raise ValueError("certainty threshold grid must be strictly increasing")
        if grid[0] < 0 or grid[-1] >= 1:
            raise ValueError("certainty thresholds must be non-negative and below one")
        if not self.load_balance_weight.is_finite() or not (
            Decimal(0) < self.load_balance_weight < Decimal(1)
        ):
            raise ValueError("load balance weight must be strictly between zero and one")
        if not self.learning_rate.is_finite() or not (
            Decimal(0) < self.learning_rate <= Decimal("0.01")
        ):
            raise ValueError("learning rate must be positive and at most 0.01")
        if not 1 <= self.max_epochs <= 30:
            raise ValueError("max epochs must be between one and thirty")
        return self

    def canonical_record(self) -> dict[str, object]:
        """Return the JSON-mode record whose canonical hash identifies this spec."""
        return self.model_dump(mode="json")

    @property
    def spec_hash(self) -> str:
        return content_sha256(self.canonical_record())


def neural_family(
    *,
    family_id: UUID,
    feature_schema_hash: str,
    audit_report_hash: str,
    h4_split_manifest_hash: str,
    h16_split_manifest_hash: str,
) -> HypothesisFamily:
    """Declare the single C.0/Phase-C hypothesis family with its fixed statement."""
    return HypothesisFamily(
        family_id=family_id,
        name=NEURAL_FAMILY_NAME,
        statement=NEURAL_FAMILY_STATEMENT,
        feature_schema_hash=feature_schema_hash,
        audit_report_hash=audit_report_hash,
        h4_split_manifest_hash=h4_split_manifest_hash,
        h16_split_manifest_hash=h16_split_manifest_hash,
    )


def neural_campaign_config_hash(
    family: HypothesisFamily,
    spec: CtmMixtureSpec,
    *,
    block_length: int,
    bootstrap_repetitions: int,
    minimum_selection_trades: int,
) -> str:
    """Bind family scope, the full spec, and the economic gate numbers into one hash."""
    for value in (block_length, bootstrap_repetitions, minimum_selection_trades):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("campaign gate numbers must be positive integers")
    return content_sha256(
        {
            "campaign_kind": "neural_residual_mixture_v1",
            "family": family.model_dump(mode="json"),
            "spec": spec.canonical_record(),
            "block_length": block_length,
            "bootstrap_repetitions": bootstrap_repetitions,
            "minimum_selection_trades": minimum_selection_trades,
        }
    )


def register_neural_campaign(
    registry_path: Path,
    family: HypothesisFamily,
    spec: CtmMixtureSpec,
    *,
    block_length: int = 96,
    bootstrap_repetitions: int = 2000,
    minimum_selection_trades: int = 200,
    revision_reason: str | None = None,
) -> HypothesisAttempt:
    """Reserve the primary or the single revision attempt for the neural family."""
    if family.name != NEURAL_FAMILY_NAME:
        raise ValueError("neural campaigns require the neural hypothesis family")
    config_hash = neural_campaign_config_hash(
        family,
        spec,
        block_length=block_length,
        bootstrap_repetitions=bootstrap_repetitions,
        minimum_selection_trades=minimum_selection_trades,
    )
    with HypothesisRegistry(registry_path) as registry:
        return registry.reserve_attempt(
            family, config_hash=config_hash, revision_reason=revision_reason
        )


def load_mixture_spec(path: Path) -> CtmMixtureSpec:
    """Load a spec whose Decimal fields are JSON strings; binary floats are rejected."""
    document = json.loads(path.read_text(encoding="utf-8"), parse_float=_reject_float)
    if not isinstance(document, dict):
        raise ValueError("mixture spec document must be an object")
    return CtmMixtureSpec.model_validate(document)


def _reject_float(text: str) -> object:
    raise ValueError(f"mixture spec numbers must be strings or integers, got {text}")
```

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/test_neural_spec.py -q -p no:cacheprovider
```

Expected: `13 passed`. If the `window_bars` parametrized case does not match `"window_bars"`, pydantic's message names the field; keep the match string as the field name.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/spec.py tests/test_neural_spec.py
git commit -m "feat: add frozen neural mixture spec and campaign registration"
```

---

### Task 8: Column layout, row tensor, membership positions, window gather

**Files:**
- Create: `src/trading_bot/neural/windows.py`
- Create: `tests/neural/test_windows.py`

**Interfaces:**
- Consumes: `TransformedFeatureMatrix` (Task 2), `TransformedViewMatrix`, `SLOPE_FIELDS`, `SPECTRAL_FIELDS` (Task 4), `ExpertKind` (Task 7).
- Produces:

```python
RATE_FIELDS = ("return_4", "return_16", "realized_volatility_16", "realized_volatility_64")
SUMMARY_FIELDS = ("range_ratio_16", "volume_zscore_32", "basis_mean_16", "basis_change_4")
MARKET_FIELD_ORDER = RATE_FIELDS + SUMMARY_FIELDS   # the 8 Phase-A fields in mixture order

@dataclass(frozen=True, slots=True)
class ColumnLayout:
    market_fields: tuple[str, ...]
    view_fields: tuple[str, ...]
    width -> int                                     # 2 * (len(market) + len(view))
    def expert_columns(self, expert: ExpertKind) -> tuple[int, ...]
    def record(self) -> dict[str, object]

@dataclass(frozen=True, slots=True)
class RowTensor:
    sample_ids: tuple[str, ...]
    decision_times_ns: tuple[int, ...]
    values: Tensor            # (N, width) float32, chronological
    layout: ColumnLayout
    def position_of(self, sample_id: str) -> int

def build_row_tensor(market: TransformedFeatureMatrix, view: TransformedViewMatrix, *,
                     decision_times_ns: tuple[int, ...], layout: ColumnLayout) -> RowTensor

@dataclass(frozen=True, slots=True)
class MembershipPositions:
    positions: Tensor                 # (M,) int64, in membership order
    sample_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]      # members without a full trailing window

def membership_positions(rows: RowTensor, member_ids: tuple[str, ...], *, window: int) -> MembershipPositions
def gather_windows(rows: RowTensor, positions: Tensor, *, window: int) -> Tensor   # (B, window, width)
```

- [ ] **Step 1: Write the failing tests**

`tests/neural/test_windows.py`:

```python
import sys
from decimal import Decimal
from importlib import import_module

import pytest
import torch

from trading_bot.feature_matrix import TransformedFeatureMatrix
from trading_bot.neural._torch import NeuralDependencyError
from trading_bot.neural.windows import (
    MARKET_FIELD_ORDER,
    ColumnLayout,
    build_row_tensor,
    gather_windows,
    membership_positions,
)
from trading_bot.view_features import VIEW_FIELDS, TransformedViewMatrix


def layout() -> ColumnLayout:
    return ColumnLayout(MARKET_FIELD_ORDER, VIEW_FIELDS)


def rows(count: int) -> tuple[TransformedFeatureMatrix, TransformedViewMatrix, tuple[int, ...]]:
    ids = tuple(f"OKX:{index}:h4" for index in range(count))
    market = TransformedFeatureMatrix(
        ids, tuple(tuple(Decimal(index) for _ in range(16)) for index in range(count))
    )
    view = TransformedViewMatrix(
        ids, tuple(tuple(Decimal(-index) for _ in range(28)) for index in range(count))
    )
    return market, view, tuple(range(1, count + 1))


def test_layout_width_and_expert_columns() -> None:
    columns = layout()

    assert columns.width == 44
    assert columns.expert_columns("rate") == tuple(range(0, 8))
    assert columns.expert_columns("summary") == tuple(range(8, 16))
    assert columns.expert_columns("slope") == tuple(range(16, 30))
    assert columns.expert_columns("spectral") == tuple(range(30, 44))
    assert columns.record()["width"] == 44


def test_row_tensor_concatenates_market_then_view_in_order() -> None:
    market, view, times = rows(5)

    tensor = build_row_tensor(market, view, decision_times_ns=times, layout=layout())

    assert tensor.values.shape == (5, 44)
    assert tensor.values[3, 0].item() == 3.0
    assert tensor.values[3, 16].item() == -3.0
    assert tensor.position_of("OKX:4:h4") == 4
    market_reordered = TransformedFeatureMatrix(market.sample_ids[::-1], market.rows)
    with pytest.raises(ValueError, match="same sample order"):
        build_row_tensor(market_reordered, view, decision_times_ns=times, layout=layout())


def test_membership_positions_drop_members_without_full_window() -> None:
    market, view, times = rows(10)
    tensor = build_row_tensor(market, view, decision_times_ns=times, layout=layout())

    members = membership_positions(
        tensor, ("OKX:1:h4", "OKX:5:h4", "OKX:9:h4"), window=4
    )

    assert members.positions.tolist() == [5, 9]
    assert members.sample_ids == ("OKX:5:h4", "OKX:9:h4")
    assert members.dropped_ids == ("OKX:1:h4",)
    with pytest.raises(ValueError, match="unknown"):
        membership_positions(tensor, ("OKX:99:h4",), window=4)
    with pytest.raises(ValueError, match="duplicate"):
        membership_positions(tensor, ("OKX:5:h4", "OKX:5:h4"), window=4)


def test_gather_windows_returns_trailing_rows_oldest_first() -> None:
    market, view, times = rows(10)
    tensor = build_row_tensor(market, view, decision_times_ns=times, layout=layout())

    windows = gather_windows(tensor, torch.tensor([5, 9]), window=3)

    assert windows.shape == (2, 3, 44)
    assert windows[0, :, 0].tolist() == [3.0, 4.0, 5.0]
    assert windows[1, :, 0].tolist() == [7.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="full window"):
        gather_windows(tensor, torch.tensor([1]), window=3)


def test_windows_module_fails_closed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "trading_bot.neural.windows", raising=False)
    with pytest.raises(NeuralDependencyError, match="uv sync --group neural"):
        import_module("trading_bot.neural.windows")
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/neural/test_windows.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.neural.windows'`.

- [ ] **Step 3: Write `neural/windows.py`**

```python
"""Causal trailing-window construction over chronologically ordered feature rows."""

from dataclasses import dataclass

from trading_bot.feature_matrix import TransformedFeatureMatrix
from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError
from trading_bot.neural.spec import ExpertKind
from trading_bot.view_features import SLOPE_FIELDS, SPECTRAL_FIELDS, TransformedViewMatrix

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error

RATE_FIELDS = ("return_4", "return_16", "realized_volatility_16", "realized_volatility_64")
SUMMARY_FIELDS = ("range_ratio_16", "volume_zscore_32", "basis_mean_16", "basis_change_4")
MARKET_FIELD_ORDER = RATE_FIELDS + SUMMARY_FIELDS
_COLUMNS_PER_FIELD = 2


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """Fixed column order: market fields then view fields, two columns per field."""

    market_fields: tuple[str, ...]
    view_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.market_fields != MARKET_FIELD_ORDER:
            raise ValueError("column layout must use the frozen market field order")
        if self.view_fields != SLOPE_FIELDS + SPECTRAL_FIELDS:
            raise ValueError("column layout must use the frozen view field order")

    @property
    def width(self) -> int:
        return _COLUMNS_PER_FIELD * (len(self.market_fields) + len(self.view_fields))

    def expert_columns(self, expert: ExpertKind) -> tuple[int, ...]:
        fields = {
            "rate": RATE_FIELDS,
            "summary": SUMMARY_FIELDS,
            "slope": SLOPE_FIELDS,
            "spectral": SPECTRAL_FIELDS,
        }[expert]
        ordered = self.market_fields + self.view_fields
        return tuple(
            column
            for field_index, field in enumerate(ordered)
            if field in fields
            for column in range(
                field_index * _COLUMNS_PER_FIELD, (field_index + 1) * _COLUMNS_PER_FIELD
            )
        )

    def record(self) -> dict[str, object]:
        return {
            "market_fields": list(self.market_fields),
            "view_fields": list(self.view_fields),
            "columns_per_field": _COLUMNS_PER_FIELD,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class RowTensor:
    """All admissible samples in chronological order as one float32 matrix."""

    sample_ids: tuple[str, ...]
    decision_times_ns: tuple[int, ...]
    values: Tensor
    layout: ColumnLayout

    def position_of(self, sample_id: str) -> int:
        try:
            return self.sample_ids.index(sample_id)
        except ValueError as error:
            raise ValueError(f"unknown sample id: {sample_id!r}") from error


def build_row_tensor(
    market: TransformedFeatureMatrix,
    view: TransformedViewMatrix,
    *,
    decision_times_ns: tuple[int, ...],
    layout: ColumnLayout,
) -> RowTensor:
    """Concatenate standardized market and view rows for identically ordered samples."""
    if market.sample_ids != view.sample_ids:
        raise ValueError("market and view matrices must share the same sample order")
    if len(decision_times_ns) != len(market.sample_ids):
        raise ValueError("decision times must align with the sample order")
    if any(later <= earlier for earlier, later in zip(decision_times_ns, decision_times_ns[1:], strict=False)):
        raise ValueError("row tensor samples must be strictly chronological")
    if len(set(market.sample_ids)) != len(market.sample_ids):
        raise ValueError("row tensor sample IDs must be unique")
    rows = [
        [float(value) for value in market_row + view_row]
        for market_row, view_row in zip(market.rows, view.rows, strict=True)
    ]
    if rows and len(rows[0]) != layout.width:
        raise ValueError("row width does not match the column layout")
    values = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, layout.width))
    return RowTensor(market.sample_ids, decision_times_ns, values, layout)


@dataclass(frozen=True, slots=True)
class MembershipPositions:
    positions: Tensor
    sample_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]


def membership_positions(
    rows: RowTensor, member_ids: tuple[str, ...], *, window: int
) -> MembershipPositions:
    """Map manifest membership IDs to row positions, dropping members lacking a full window."""
    if window < 1:
        raise ValueError("window must be positive")
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("membership contains duplicate sample IDs")
    index = {sample_id: position for position, sample_id in enumerate(rows.sample_ids)}
    kept: list[int] = []
    kept_ids: list[str] = []
    dropped: list[str] = []
    for sample_id in member_ids:
        position = index.get(sample_id)
        if position is None:
            raise ValueError(f"unknown membership sample id: {sample_id!r}")
        if position < window - 1:
            dropped.append(sample_id)
        else:
            kept.append(position)
            kept_ids.append(sample_id)
    return MembershipPositions(
        torch.tensor(kept, dtype=torch.int64), tuple(kept_ids), tuple(dropped)
    )


def gather_windows(rows: RowTensor, positions: Tensor, *, window: int) -> Tensor:
    """Return the trailing `window` rows ending at each position, oldest first."""
    if positions.dim() != 1:
        raise ValueError("positions must be a one-dimensional index tensor")
    if positions.numel() and int(positions.min().item()) < window - 1:
        raise ValueError("every position must have a full window of preceding rows")
    offsets = torch.arange(window - 1, -1, -1, dtype=torch.int64, device=positions.device)
    index = positions[:, None] - offsets[None, :]
    return rows.values[index]
```

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/neural/test_windows.py -q -p no:cacheprovider
```

Expected: `5 passed`.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/windows.py tests/neural/test_windows.py
git commit -m "feat: add causal window tensors for neural challengers"
```

---

### Task 9: Residual experts, GRU encoder, mixture, certainty, loss

**Files:**
- Create: `src/trading_bot/neural/mixture.py`
- Create: `tests/neural/test_mixture.py`

**Interfaces:**
- Consumes: `ColumnLayout` (Task 8), `EXPERT_ORDER`, `GateKind` (Task 7).
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EncoderOutput:
    residual: Tensor              # (B,)
    gate_logits: Tensor           # (B, K + 1)
    certainty_trace: Tensor | None   # (B, T) for tick-based encoders; None for GRU
    ticks_used: Tensor | None        # (B,) int64 or None

class Encoder(Protocol):
    def __call__(self, windows: Tensor) -> EncoderOutput: ...

class ResidualExpert(nn.Module):      # __init__(columns: tuple[int, ...], hidden: int)
class GruEncoder(nn.Module):          # __init__(feature_width: int, hidden: int, expert_count: int)
class ResidualMixture(nn.Module):
    def __init__(self, layout: ColumnLayout, *, expert_hidden: int, gate_kind: GateKind,
                 encoder: nn.Module | None, selected_expert: int | None = None) -> None
    def forward(self, current_rows: Tensor, windows: Tensor | None, base_logit: Tensor) -> MixtureOutput

@dataclass(frozen=True, slots=True)
class MixtureOutput:
    logit: Tensor; probability: Tensor; gate_weights: Tensor; expert_residuals: Tensor
    encoder_residual: Tensor | None; certainty_trace: Tensor | None; ticks_used: Tensor | None

def binary_certainty(probability: Tensor) -> Tensor
def mixture_loss(output: MixtureOutput, labels: Tensor, *, load_balance_weight: float) -> tuple[Tensor, Tensor, Tensor]
```

  Plan 2's `CtmEncoder` returns the same `EncoderOutput` with a populated `certainty_trace`, and the loss module applies the dual-tick rule only when the trace is present.

- [ ] **Step 1: Write the failing tests**

`tests/neural/test_mixture.py`:

```python
import pytest
import torch

from trading_bot.neural._torch import seed_everything
from trading_bot.neural.mixture import (
    GruEncoder,
    ResidualExpert,
    ResidualMixture,
    binary_certainty,
    mixture_loss,
)
from trading_bot.neural.windows import MARKET_FIELD_ORDER, ColumnLayout
from trading_bot.view_features import VIEW_FIELDS

LAYOUT = ColumnLayout(MARKET_FIELD_ORDER, VIEW_FIELDS)


def batch(size: int = 6, window: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_everything(1)
    rows = torch.randn(size, LAYOUT.width)
    windows = torch.randn(size, window, LAYOUT.width)
    base = torch.linspace(-1.0, 1.0, size)
    return rows, windows, base


def test_residual_expert_starts_at_zero_and_reads_only_its_columns() -> None:
    expert = ResidualExpert(LAYOUT.expert_columns("rate"), hidden=8)
    rows, _, _ = batch()

    assert torch.equal(expert(rows), torch.zeros(6))
    last = expert.body[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.fill_(1.0)
    changed = rows.clone()
    changed[:, 20] += 100.0
    assert torch.allclose(expert(rows), expert(changed))


def test_learned_mixture_is_identity_at_init_and_gate_sums_to_one() -> None:
    encoder = GruEncoder(LAYOUT.width, hidden=16, expert_count=4)
    model = ResidualMixture(LAYOUT, expert_hidden=8, gate_kind="learned", encoder=encoder)
    rows, windows, base = batch()

    output = model(rows, windows, base)

    assert torch.allclose(output.logit, base)
    assert output.gate_weights.shape == (6, 5)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(6))
    assert torch.allclose(output.gate_weights, torch.full((6, 5), 0.2))
    assert output.encoder_residual is not None and output.certainty_trace is None


@pytest.mark.parametrize(
    "gate_kind, selected, expected",
    (
        ("none", None, [0.0, 0.0, 0.0, 0.0]),
        ("uniform", None, [0.25, 0.25, 0.25, 0.25]),
        ("best_single", 2, [0.0, 0.0, 1.0, 0.0]),
    ),
)
def test_encoder_free_gates_use_fixed_weights(
    gate_kind: str, selected: int | None, expected: list[float]
) -> None:
    model = ResidualMixture(
        LAYOUT, expert_hidden=8, gate_kind=gate_kind, encoder=None, selected_expert=selected
    )
    rows, _, base = batch()

    output = model(rows, None, base)

    assert output.gate_weights[0].tolist() == expected
    assert output.encoder_residual is None
    assert torch.allclose(output.logit, base)


def test_mixture_rejects_inconsistent_gate_and_encoder() -> None:
    with pytest.raises(ValueError, match="learned gate requires an encoder"):
        ResidualMixture(LAYOUT, expert_hidden=8, gate_kind="learned", encoder=None)
    with pytest.raises(ValueError, match="best_single requires"):
        ResidualMixture(LAYOUT, expert_hidden=8, gate_kind="best_single", encoder=None)
    with pytest.raises(ValueError, match="does not accept an encoder"):
        ResidualMixture(
            LAYOUT, expert_hidden=8, gate_kind="uniform",
            encoder=GruEncoder(LAYOUT.width, hidden=4, expert_count=4),
        )


def test_certainty_is_zero_at_half_and_one_at_the_limits() -> None:
    certainty = binary_certainty(torch.tensor([0.5, 0.0, 1.0, 0.9]))

    assert torch.allclose(certainty[:3], torch.tensor([0.0, 1.0, 1.0]), atol=1e-6)
    assert 0.0 < certainty[3].item() < 1.0


def test_loss_penalizes_gate_imbalance_and_backpropagates_to_experts() -> None:
    encoder = GruEncoder(LAYOUT.width, hidden=16, expert_count=4)
    model = ResidualMixture(LAYOUT, expert_hidden=8, gate_kind="learned", encoder=encoder)
    rows, windows, base = batch()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    output = model(rows, windows, base)
    total, bce, balance = mixture_loss(output, labels, load_balance_weight=0.01)
    total.backward()

    assert torch.isclose(balance, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(total, bce, atol=1e-6)
    first_expert = model.experts[0]
    assert isinstance(first_expert, ResidualExpert)
    last = first_expert.body[-1]
    assert isinstance(last, torch.nn.Linear) and last.weight.grad is not None
    assert encoder.gate_head.weight.grad is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/neural/test_mixture.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.neural.mixture'`.

- [ ] **Step 3: Write `neural/mixture.py`**

```python
"""Regime-gated residual mixture: shallow experts, a sequence encoder gate, frozen base."""

import math
from dataclasses import dataclass
from typing import Protocol

from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError
from trading_bot.neural.spec import EXPERT_ORDER, GateKind
from trading_bot.neural.windows import ColumnLayout

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error

_EPSILON = 1e-7


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    residual: Tensor
    gate_logits: Tensor
    certainty_trace: Tensor | None
    ticks_used: Tensor | None


class Encoder(Protocol):
    def __call__(self, windows: Tensor) -> EncoderOutput: ...


@dataclass(frozen=True, slots=True)
class MixtureOutput:
    logit: Tensor
    probability: Tensor
    gate_weights: Tensor
    expert_residuals: Tensor
    encoder_residual: Tensor | None
    certainty_trace: Tensor | None
    ticks_used: Tensor | None


class ResidualExpert(nn.Module):
    """A tiny head over one view's columns whose output starts at exactly zero."""

    def __init__(self, columns: tuple[int, ...], hidden: int) -> None:
        super().__init__()
        if not columns or hidden < 1:
            raise ValueError("residual expert requires columns and a positive hidden width")
        self.columns: Tensor
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.int64))
        self.body = nn.Sequential(nn.Linear(len(columns), hidden), nn.GELU(), nn.Linear(hidden, 1))
        _zero_init(self.body[-1])

    def forward(self, rows: Tensor) -> Tensor:
        selected = rows.index_select(1, self.columns)
        return self.body(selected).squeeze(-1)


class GruEncoder(nn.Module):
    """Spec 6.1 comparator: a one-layer GRU with the same residual and gate heads as the CTM."""

    def __init__(self, feature_width: int, hidden: int, expert_count: int) -> None:
        super().__init__()
        if feature_width < 1 or hidden < 1 or expert_count < 1:
            raise ValueError("gru encoder dimensions must be positive")
        self.gru = nn.GRU(feature_width, hidden, batch_first=True)
        self.residual_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Linear(hidden, expert_count + 1)
        _zero_init(self.residual_head)
        _zero_init(self.gate_head)

    def forward(self, windows: Tensor) -> EncoderOutput:
        _, hidden = self.gru(windows)
        state = hidden[-1]
        return EncoderOutput(
            residual=self.residual_head(state).squeeze(-1),
            gate_logits=self.gate_head(state),
            certainty_trace=None,
            ticks_used=None,
        )


class ResidualMixture(nn.Module):
    """`logit = base + Σ π_k r_k` over experts (and the encoder residual when learned)."""

    def __init__(
        self,
        layout: ColumnLayout,
        *,
        expert_hidden: int,
        gate_kind: GateKind,
        encoder: nn.Module | None,
        selected_expert: int | None = None,
    ) -> None:
        super().__init__()
        if gate_kind == "learned" and encoder is None:
            raise ValueError("learned gate requires an encoder")
        if gate_kind != "learned" and encoder is not None:
            raise ValueError(f"gate kind {gate_kind!r} does not accept an encoder")
        if gate_kind == "best_single" and (
            selected_expert is None or not 0 <= selected_expert < len(EXPERT_ORDER)
        ):
            raise ValueError("best_single requires a selected expert index")
        if gate_kind != "best_single" and selected_expert is not None:
            raise ValueError("selected expert is only valid for best_single gating")
        self.layout = layout
        self.expert_hidden = expert_hidden
        self.gate_kind: GateKind = gate_kind
        self.selected_expert = selected_expert
        self.experts = nn.ModuleList(
            ResidualExpert(layout.expert_columns(expert), expert_hidden) for expert in EXPERT_ORDER
        )
        self.encoder = encoder

    def forward(
        self, current_rows: Tensor, windows: Tensor | None, base_logit: Tensor
    ) -> MixtureOutput:
        residuals = torch.stack([expert(current_rows) for expert in self.experts], dim=1)
        batch = residuals.shape[0]
        expert_count = residuals.shape[1]
        if self.encoder is not None:
            if windows is None:
                raise ValueError("learned gating requires windows")
            encoded = self.encoder(windows)
            weights = torch.softmax(encoded.gate_logits, dim=-1)
            combined = torch.cat([residuals, encoded.residual[:, None]], dim=1)
            encoder_residual: Tensor | None = encoded.residual
            trace, ticks = encoded.certainty_trace, encoded.ticks_used
        else:
            weights = self._fixed_weights(batch, expert_count, residuals)
            combined = residuals
            encoder_residual, trace, ticks = None, None, None
        logit = base_logit + (weights * combined).sum(dim=1)
        return MixtureOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            gate_weights=weights,
            expert_residuals=residuals,
            encoder_residual=encoder_residual,
            certainty_trace=trace,
            ticks_used=ticks,
        )

    def _fixed_weights(self, batch: int, expert_count: int, like: Tensor) -> Tensor:
        if self.gate_kind == "none":
            return torch.zeros((batch, expert_count), dtype=like.dtype, device=like.device)
        if self.gate_kind == "uniform":
            return torch.full(
                (batch, expert_count), 1.0 / expert_count, dtype=like.dtype, device=like.device
            )
        weights = torch.zeros((batch, expert_count), dtype=like.dtype, device=like.device)
        if self.selected_expert is None:
            raise ValueError("best_single requires a selected expert index")
        weights[:, self.selected_expert] = 1.0
        return weights


def binary_certainty(probability: Tensor) -> Tensor:
    """`1 - H(p) / ln 2`: zero at a coin flip, one at a certain forecast."""
    clamped = probability.clamp(_EPSILON, 1.0 - _EPSILON)
    entropy = -(clamped * clamped.log() + (1.0 - clamped) * (1.0 - clamped).log())
    return 1.0 - entropy / math.log(2.0)


def mixture_loss(
    output: MixtureOutput, labels: Tensor, *, load_balance_weight: float
) -> tuple[Tensor, Tensor, Tensor]:
    """BCE on the combined logit plus the squared coefficient of variation of gate usage."""
    if load_balance_weight < 0:
        raise ValueError("load balance weight must be non-negative")
    bce = F.binary_cross_entropy_with_logits(output.logit, labels)
    mean_usage = output.gate_weights.mean(dim=0)
    balance = mean_usage.var(unbiased=False) / (mean_usage.mean() ** 2 + _EPSILON)
    return bce + load_balance_weight * balance, bce, balance


def _zero_init(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
```

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/neural/test_mixture.py -q -p no:cacheprovider
```

Expected: `8 passed`.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/mixture.py tests/neural/test_mixture.py
git commit -m "feat: add regime-gated residual mixture with gru encoder"
```

### Task 10: Deterministic training loop and compute evidence

**Files:**
- Create: `src/trading_bot/neural/training.py`
- Create: `src/trading_bot/neural/compute_evidence.py`
- Create: `tests/neural/test_training.py`
- Create: `tests/neural/test_compute_evidence.py`

**Interfaces:**
- Consumes: `RowTensor`, `gather_windows` (Task 8); `ResidualMixture`, `mixture_loss` (Task 9); `seed_everything` (Task 1).
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TrainingData:
    rows: RowTensor
    base_logits: Tensor            # (N,) float32, aligned with rows
    labels: Tensor                 # (N,) float32 in {0, 1}
    train_positions: Tensor        # (M,) int64
    calibration_positions: Tensor  # (C,) int64
    window: int

@dataclass(frozen=True, slots=True)
class TrainingRecord:
    epochs_run: int
    best_epoch: int
    best_calibration_loss: Decimal
    stopped_early: bool
    training_seconds: Decimal
    def record(self) -> dict[str, object]

def evaluate_loss(model, data, positions, *, batch_size, load_balance_weight) -> float
def train_mixture(model, data, *, batch_size, learning_rate, max_epochs, patience,
                  load_balance_weight, seed, device) -> TrainingRecord

@dataclass(frozen=True, slots=True)
class ComputeEvidence:
    parameter_count: int
    peak_vram_bytes: int
    training_seconds: Decimal
    inference_p50_ms: Decimal
    inference_p95_ms: Decimal
    gpu_seconds: Decimal
    def record(self) -> dict[str, object]

def count_parameters(model: nn.Module) -> int
def peak_vram_bytes() -> int
def measure_inference_latency(forward: Callable[[], object], *, repetitions: int) -> tuple[Decimal, Decimal]
def gpu_seconds(training_seconds: Decimal, *, device: str) -> Decimal
```

- [ ] **Step 1: Write the failing tests**

`tests/neural/test_training.py`:

```python
from decimal import Decimal

import torch

from trading_bot.neural._torch import seed_everything
from trading_bot.neural.mixture import GruEncoder, ResidualMixture
from trading_bot.neural.training import TrainingData, evaluate_loss, train_mixture
from trading_bot.neural.windows import MARKET_FIELD_ORDER, ColumnLayout, RowTensor
from trading_bot.view_features import VIEW_FIELDS

LAYOUT = ColumnLayout(MARKET_FIELD_ORDER, VIEW_FIELDS)
WINDOW = 4


def separable_data(count: int = 240) -> TrainingData:
    seed_everything(3)
    values = torch.randn(count, LAYOUT.width)
    labels = (values[:, 0] > 0).float()
    rows = RowTensor(
        tuple(f"s{index}" for index in range(count)),
        tuple(range(1, count + 1)),
        values,
        LAYOUT,
    )
    positions = torch.arange(WINDOW - 1, count)
    split = int(positions.numel() * 0.75)
    return TrainingData(
        rows=rows,
        base_logits=torch.zeros(count),
        labels=labels,
        train_positions=positions[:split],
        calibration_positions=positions[split:],
        window=WINDOW,
    )


def learned_model() -> ResidualMixture:
    return ResidualMixture(
        LAYOUT,
        expert_hidden=8,
        gate_kind="learned",
        encoder=GruEncoder(LAYOUT.width, hidden=8, expert_count=4),
    )


def test_training_lowers_calibration_loss_and_reports_record() -> None:
    data = separable_data()
    model = learned_model()
    before = evaluate_loss(
        model, data, data.calibration_positions, batch_size=64, load_balance_weight=0.01
    )

    record = train_mixture(
        model, data, batch_size=64, learning_rate=Decimal("0.01"), max_epochs=6,
        patience=5, load_balance_weight=Decimal("0.01"), seed=5, device="cpu",
    )
    after = evaluate_loss(
        model, data, data.calibration_positions, batch_size=64, load_balance_weight=0.01
    )

    assert after < before
    assert 1 <= record.epochs_run <= 6
    assert record.best_calibration_loss == Decimal(str(after))
    assert record.record()["stopped_early"] is False
    assert record.training_seconds >= 0


def test_training_is_bitwise_reproducible_for_the_same_seed() -> None:
    data = separable_data()
    first = learned_model()
    second = learned_model()

    for model in (first, second):
        train_mixture(
            model, data, batch_size=64, learning_rate=Decimal("0.01"), max_epochs=3,
            patience=5, load_balance_weight=Decimal("0.01"), seed=9, device="cpu",
        )

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_training_stops_early_when_nothing_improves() -> None:
    data = separable_data()
    model = ResidualMixture(LAYOUT, expert_hidden=8, gate_kind="none", encoder=None)

    record = train_mixture(
        model, data, batch_size=64, learning_rate=Decimal("0.01"), max_epochs=30,
        patience=2, load_balance_weight=Decimal("0.01"), seed=1, device="cpu",
    )

    assert record.stopped_early is True
    assert record.epochs_run == 2
    assert record.best_epoch == 0
```

`tests/neural/test_compute_evidence.py`:

```python
from decimal import Decimal

import torch

from trading_bot.neural.compute_evidence import (
    ComputeEvidence,
    count_parameters,
    gpu_seconds,
    measure_inference_latency,
    peak_vram_bytes,
)


def test_parameter_count_and_vram_are_non_negative_integers() -> None:
    model = torch.nn.Linear(3, 2)

    assert count_parameters(model) == 8
    assert peak_vram_bytes() >= 0


def test_latency_percentiles_are_ordered_decimals() -> None:
    calls = 0

    def forward() -> None:
        nonlocal calls
        calls += 1

    p50, p95 = measure_inference_latency(forward, repetitions=20)

    assert calls == 20 + 1
    assert isinstance(p50, Decimal) and isinstance(p95, Decimal)
    assert Decimal(0) <= p50 <= p95


def test_gpu_seconds_and_record_shape() -> None:
    evidence = ComputeEvidence(8, 0, Decimal("1.5"), Decimal("0.1"), Decimal("0.2"), Decimal(0))

    assert gpu_seconds(Decimal("1.5"), device="cpu") == Decimal(0)
    assert gpu_seconds(Decimal("1.5"), device="cuda") == Decimal("1.5")
    assert evidence.record() == {
        "parameter_count": 8,
        "peak_vram_bytes": 0,
        "training_seconds": Decimal("1.5"),
        "inference_p50_ms": Decimal("0.1"),
        "inference_p95_ms": Decimal("0.2"),
        "gpu_seconds": Decimal(0),
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/neural/test_training.py tests/neural/test_compute_evidence.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError` for both modules.

- [ ] **Step 3: Write `neural/training.py`**

```python
"""Deterministic mini-batch training with calibration-half early stopping."""

import copy
import time
from dataclasses import dataclass
from decimal import Decimal

from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError, seed_everything
from trading_bot.neural.mixture import ResidualMixture, mixture_loss
from trading_bot.neural.windows import RowTensor, gather_windows

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error

_IMPROVEMENT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class TrainingData:
    rows: RowTensor
    base_logits: Tensor
    labels: Tensor
    train_positions: Tensor
    calibration_positions: Tensor
    window: int

    def __post_init__(self) -> None:
        count = self.rows.values.shape[0]
        if self.base_logits.shape != (count,) or self.labels.shape != (count,):
            raise ValueError("base logits and labels must align with the row tensor")
        if self.train_positions.numel() == 0 or self.calibration_positions.numel() == 0:
            raise ValueError("training and calibration positions must not be empty")
        if self.window < 1:
            raise ValueError("window must be positive")


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    epochs_run: int
    best_epoch: int
    best_calibration_loss: Decimal
    stopped_early: bool
    training_seconds: Decimal

    def record(self) -> dict[str, object]:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_calibration_loss": self.best_calibration_loss,
            "stopped_early": self.stopped_early,
            "training_seconds": self.training_seconds,
        }


def evaluate_loss(
    model: ResidualMixture,
    data: TrainingData,
    positions: Tensor,
    *,
    batch_size: int,
    load_balance_weight: float,
) -> float:
    """Mean total loss over the given positions without gradient tracking."""
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, positions.numel(), batch_size):
            batch = positions[start : start + batch_size]
            output = model(*_inputs(model, data, batch, device))
            loss, _, _ = mixture_loss(
                output, data.labels[batch].to(device), load_balance_weight=load_balance_weight
            )
            total += float(loss.item()) * int(batch.numel())
            count += int(batch.numel())
    return total / count


def train_mixture(
    model: ResidualMixture,
    data: TrainingData,
    *,
    batch_size: int,
    learning_rate: Decimal,
    max_epochs: int,
    patience: int,
    load_balance_weight: Decimal,
    seed: int,
    device: str,
) -> TrainingRecord:
    """AdamW with seeded shuffling; keeps the best calibration-half state."""
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("batch size, epochs, and patience must be positive")
    seed_everything(seed)
    target = torch.device(device)
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(seed)
    balance = float(load_balance_weight)
    started = time.perf_counter()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = evaluate_loss(
        model, data, data.calibration_positions, batch_size=batch_size,
        load_balance_weight=balance,
    )
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(max_epochs):
        model.train()
        order = data.train_positions[torch.randperm(data.train_positions.numel(), generator=generator)]
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            output = model(*_inputs(model, data, batch, target))
            loss, _, _ = mixture_loss(
                output, data.labels[batch].to(target), load_balance_weight=balance
            )
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        calibration_loss = evaluate_loss(
            model, data, data.calibration_positions, batch_size=batch_size,
            load_balance_weight=balance,
        )
        if calibration_loss < best_loss - _IMPROVEMENT_TOLERANCE:
            best_loss = calibration_loss
            best_epoch = epochs_run
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainingRecord(
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_calibration_loss=Decimal(str(best_loss)),
        stopped_early=stopped_early,
        training_seconds=Decimal(str(round(time.perf_counter() - started, 6))),
    )


def _inputs(
    model: ResidualMixture, data: TrainingData, batch: Tensor, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor]:
    current = data.rows.values[batch].to(device)
    windows = (
        gather_windows(data.rows, batch, window=data.window).to(device)
        if model.encoder is not None
        else None
    )
    return current, windows, data.base_logits[batch].to(device)
```

Note for the early-stopping test: `best_epoch == 0` means the untrained state (evaluated before epoch 1) was never beaten; `patience=2` tolerates two consecutive non-improving epochs and stops after the second (`epochs_run == 2`). With `gate_kind="none"` the expert outputs carry zero weight, so the loss cannot change and every epoch is non-improving.

- [ ] **Step 4: Write `neural/compute_evidence.py`**

```python
"""Protocol section 10 compute evidence: parameters, VRAM, latency, runtime proxy."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error


@dataclass(frozen=True, slots=True)
class ComputeEvidence:
    parameter_count: int
    peak_vram_bytes: int
    training_seconds: Decimal
    inference_p50_ms: Decimal
    inference_p95_ms: Decimal
    gpu_seconds: Decimal

    def __post_init__(self) -> None:
        if self.parameter_count < 0 or self.peak_vram_bytes < 0:
            raise ValueError("compute counts must be non-negative")
        for value in (
            self.training_seconds, self.inference_p50_ms, self.inference_p95_ms, self.gpu_seconds
        ):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError("compute timings must be finite non-negative Decimals")

    def record(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count,
            "peak_vram_bytes": self.peak_vram_bytes,
            "training_seconds": self.training_seconds,
            "inference_p50_ms": self.inference_p50_ms,
            "inference_p95_ms": self.inference_p95_ms,
            "gpu_seconds": self.gpu_seconds,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def peak_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def measure_inference_latency(
    forward: Callable[[], object], *, repetitions: int
) -> tuple[Decimal, Decimal]:
    """Wall-clock p50 and p95 in milliseconds after one warm-up call."""
    if repetitions < 1:
        raise ValueError("latency repetitions must be positive")
    forward()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        forward()
        samples.append((time.perf_counter() - started) * 1_000.0)
    samples.sort()
    return _nearest_rank(samples, Decimal("0.50")), _nearest_rank(samples, Decimal("0.95"))


def gpu_seconds(training_seconds: Decimal, *, device: str) -> Decimal:
    """Energy/runtime proxy: training seconds when a GPU was used, else zero."""
    return training_seconds if device.startswith("cuda") else Decimal(0)


def _nearest_rank(sorted_values: list[float], probability: Decimal) -> Decimal:
    rank = int((probability * Decimal(len(sorted_values))).to_integral_value(ROUND_CEILING))
    index = min(max(rank, 1), len(sorted_values)) - 1
    return Decimal(str(round(sorted_values[index], 6)))
```

- [ ] **Step 5: Run the tests**

```powershell
uv run pytest tests/neural/test_training.py tests/neural/test_compute_evidence.py -q -p no:cacheprovider
```

Expected: `6 passed`.

- [ ] **Step 6: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/training.py src/trading_bot/neural/compute_evidence.py tests/neural/test_training.py tests/neural/test_compute_evidence.py
git commit -m "feat: add deterministic training loop and compute evidence"
```

---

### Task 11: Registered neural fold runner (`gru_moe` first)

**Files:**
- Create: `src/trading_bot/neural/fold_run.py`
- Create: `tests/neural/test_fold_run.py`

**Interfaces:**
- Consumes: `verify_research_inputs`, `ResearchInputs`, `PhaseAFoldError`, and the private Phase-A helpers listed in Task 6; `build_enriched_samples`, `load_research_bars`, `fit_feature_matrix`, `transform_feature_matrix`; `build_view_samples`, `fit_view_matrix`, `transform_view_matrix`, `c0_view_feature_schema`; `fit_logistic_rows`, `fit_platt_calibrator_rows`; `fit_residual_interval`; `HypothesisRegistry`, `NeuralFoldRun`; `CtmMixtureSpec`, `neural_campaign_config_hash`, `NEURAL_FAMILY_NAME`; Tasks 8–10 modules.
- Produces:

```python
class NeuralFoldError(PhaseAFoldError): ...      # same `error_code` attribute

@dataclass(frozen=True, slots=True)
class NeuralFoldConfig:
    capture_root: Path
    split_manifest_path: Path
    audit_path: Path
    registry_path: Path
    output_path: Path
    checkpoint_path: Path
    hypothesis_family_id: UUID
    attempt_id: UUID
    spec: CtmMixtureSpec
    encoder_kind: EncoderKind
    gate_kind: GateKind
    seed_ordinal: Literal[0, 1, 2]
    fold_index: int
    device: Literal["cpu", "cuda"] = "cpu"
    block_length: int = 96
    bootstrap_repetitions: int = 2000
    minimum_selection_trades: int = 200
    random_seed -> int (property: spec.seeds[seed_ordinal])

@dataclass(frozen=True, slots=True)
class NeuralFoldArtifact:
    output_path: Path; checkpoint_path: Path; report_hash: str; checkpoint_hash: str
    attempt_id: UUID; horizon_bars: Literal[4, 16]; encoder_kind: EncoderKind
    gate_kind: GateKind; seed_ordinal: Literal[0, 1, 2]; fold_index: Literal[0, 1, 2]

def run_neural_fold(config: NeuralFoldConfig) -> NeuralFoldArtifact
```

  Error codes added by this runner: `FAMILY_NOT_NEURAL`, `VIEW_SCHEMA_HASH_MISMATCH`, `SPEC_SCHEMA_HASH_MISMATCH`, `ENCODER_NOT_AVAILABLE`, `TEST_WINDOW_UNAVAILABLE`, `CHECKPOINT_ALREADY_EXISTS`, `LOGISTIC_CLASS_MISSING`, `INSUFFICIENT_SELECTION_TRADES` (reused), `VRAM_PREFLIGHT_FAILED`.

- Task 12 adds `verify_neural_fold` and the encoder-free comparators' tests; the runner written here already supports every `gate_kind`.

- [ ] **Step 1: Write the failing test and fixture**

`tests/neural/test_fold_run.py`:

```python
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from tests.test_phase_a_audit import complete_cost_capture
from tests.test_phase_a_fold_run import FIRST_OPEN_NS, INTERVAL_NS, _directional_capture
from trading_bot.hypothesis_registry import HypothesisRegistry
from trading_bot.market_features import phase_a_market_feature_schema
from trading_bot.neural.fold_run import NeuralFoldConfig, NeuralFoldError, run_neural_fold
from trading_bot.neural.spec import (
    CtmMixtureSpec,
    EncoderKind,
    GateKind,
    neural_family,
    register_neural_campaign,
)
from trading_bot.phase_a_audit import build_phase_a_audit
from trading_bot.splits import WalkForwardConfig
from trading_bot.view_features import c0_view_feature_schema
from trading_bot.walk_forward_run import run_capture_walk_forward

FAMILY_ID = UUID("00000000-0000-0000-0000-000000000901")


def test_spec(**overrides: object) -> CtmMixtureSpec:
    values: dict[str, object] = {
        "version": "1.0.0",
        "reference_commit": "0" * 40,
        "feature_schema_hash": phase_a_market_feature_schema().schema_hash,
        "view_schema_hash": c0_view_feature_schema().schema_hash,
        "certainty_threshold_grid": (Decimal(0), Decimal("0.02")),
        "seeds": (31, 32, 33),
        "max_epochs": 2,
    }
    values.update(overrides)
    return CtmMixtureSpec.model_validate(values)


def neural_fold_fixture(
    tmp_path: Path,
    *,
    encoder_kind: EncoderKind = "gru",
    gate_kind: GateKind = "learned",
    spec: CtmMixtureSpec | None = None,
) -> NeuralFoldConfig:
    capture_root = _directional_capture(tmp_path)
    split_path = tmp_path / "h4-split.json"
    first_decision_ns = FIRST_OPEN_NS + 2 * INTERVAL_NS
    split = run_capture_walk_forward(
        capture_root,
        output_path=split_path,
        config=WalkForwardConfig(
            train_duration_ns=600 * INTERVAL_NS,
            validation_duration_ns=300 * INTERVAL_NS,
            test_duration_ns=300 * INTERVAL_NS,
            step_ns=250 * INTERVAL_NS,
            embargo_ns=4 * INTERVAL_NS,
            final_holdout_start_ns=first_decision_ns + 1_900 * INTERVAL_NS,
        ),
        horizon_bars=4,
    )
    audit_path = tmp_path / "phase-a-audit.json"
    audit = build_phase_a_audit(capture_root, complete_cost_capture(tmp_path), audit_path)
    resolved = test_spec() if spec is None else spec
    family = neural_family(
        family_id=FAMILY_ID,
        feature_schema_hash=phase_a_market_feature_schema().schema_hash,
        audit_report_hash=audit.report_hash,
        h4_split_manifest_hash=split.manifest_hash,
        h16_split_manifest_hash="1" * 64,
    )
    registry_path = tmp_path / "neural.sqlite3"
    attempt = register_neural_campaign(
        registry_path, family, resolved,
        block_length=1, bootstrap_repetitions=20, minimum_selection_trades=1,
    )
    return NeuralFoldConfig(
        capture_root=capture_root,
        split_manifest_path=split_path,
        audit_path=audit_path,
        registry_path=registry_path,
        output_path=tmp_path / "neural-fold.json",
        checkpoint_path=tmp_path / "neural-fold.pt",
        hypothesis_family_id=FAMILY_ID,
        attempt_id=attempt.attempt_id,
        spec=resolved,
        encoder_kind=encoder_kind,
        gate_kind=gate_kind,
        seed_ordinal=0,
        fold_index=0,
        block_length=1,
        bootstrap_repetitions=20,
        minimum_selection_trades=1,
    )


def test_gru_moe_fold_publishes_immutable_report_checkpoint_and_ledger(tmp_path: Path) -> None:
    config = neural_fold_fixture(tmp_path)

    artifact = run_neural_fold(config)
    document = json.loads(config.output_path.read_text(encoding="utf-8"))

    assert artifact.report_hash == document["report_hash"]
    assert document["encoder_kind"] == "gru" and document["gate_kind"] == "learned"
    assert document["random_seed"] == 31 and document["seed_ordinal"] == 0
    assert document["horizon_bars"] == 4 and document["fold_index"] == 0
    assert document["final_holdout_read_count"] == 0
    assert document["test_sample_count"] == len(document["test_signals"])
    assert len(document["test_gate_weights"][0]) == 5
    assert all(
        abs(sum(Decimal(weight) for weight in weights) - Decimal(1)) < Decimal("1e-5")
        for weights in document["test_gate_weights"]
    )
    assert set(document["test_signals"]) <= {-1, 0, 1}
    assert document["compute_evidence"]["parameter_count"] > 0
    assert document["training"]["epochs_run"] <= 2
    assert document["checkpoint_sha256"] == hashlib.sha256(
        config.checkpoint_path.read_bytes()
    ).hexdigest()
    assert document["window_unavailable_train_ids"]
    assert document["spec"]["spec_hash"] == config.spec.spec_hash
    with HypothesisRegistry(config.registry_path) as registry:
        runs = registry.list_neural_fold_runs()
    assert len(runs) == 1 and runs[0].status == "completed"
    assert runs[0].artifact_hash == artifact.report_hash

    with pytest.raises(NeuralFoldError) as raised:
        run_neural_fold(config)
    assert raised.value.error_code == "OUTPUT_ALREADY_EXISTS"


def test_neural_fold_fails_closed_on_mismatched_spec_hashes_and_records_failure(
    tmp_path: Path,
) -> None:
    config = neural_fold_fixture(tmp_path)
    wrong = replace(config, spec=test_spec(view_schema_hash="f" * 64))

    with pytest.raises(NeuralFoldError) as raised:
        run_neural_fold(wrong)

    assert raised.value.error_code == "ATTEMPT_CONFIG_HASH_MISMATCH"
    assert not config.output_path.exists()


def test_neural_fold_rejects_ctm_encoder_until_plan_two(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ENCODER_NOT_AVAILABLE"):
        neural_fold_fixture(tmp_path, encoder_kind="ctm", gate_kind="learned")


def test_neural_fold_config_rejects_inconsistent_gate_and_encoder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="learned gate requires"):
        neural_fold_fixture(tmp_path, encoder_kind="none", gate_kind="learned")
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/neural/test_fold_run.py -q -p no:cacheprovider
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trading_bot.neural.fold_run'`.

- [ ] **Step 3: Write `neural/fold_run.py`**

```python
"""Registered, leakage-safe fold runner for the C.0 residual mixture campaign."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from time import time_ns
from typing import Literal
from uuid import UUID

from trading_bot.abstention import AbstentionDecision
from trading_bot.canonical import content_sha256
from trading_bot.evaluation import EvaluationResult, evaluate_signals
from trading_bot.feature_matrix import fit_feature_matrix, transform_feature_matrix
from trading_bot.hypothesis_registry import (
    HypothesisRegistry,
    NeuralFoldRun,
    RegistryConflictError,
)
from trading_bot.logistic_baseline import (
    LogisticVectorModel,
    PlattCalibrator,
    fit_logistic_rows,
    fit_platt_calibrator_rows,
)
from trading_bot.market_features import EnrichedBarSample, build_enriched_samples
from trading_bot.neural._torch import INSTALL_HINT, NeuralDependencyError, seed_everything
from trading_bot.neural.compute_evidence import (
    ComputeEvidence,
    count_parameters,
    gpu_seconds,
    measure_inference_latency,
    peak_vram_bytes,
)
from trading_bot.neural.mixture import GruEncoder, ResidualMixture, binary_certainty
from trading_bot.neural.spec import (
    EXPERT_ORDER,
    NEURAL_FAMILY_NAME,
    CtmMixtureSpec,
    EncoderKind,
    GateKind,
    neural_campaign_config_hash,
)
from trading_bot.neural.training import TrainingData, TrainingRecord, evaluate_loss, train_mixture
from trading_bot.neural.windows import (
    MARKET_FIELD_ORDER,
    ColumnLayout,
    MembershipPositions,
    RowTensor,
    build_row_tensor,
    gather_windows,
    membership_positions,
)
from trading_bot.phase_a_fold_run import (
    PhaseAFoldError,
    ResearchInputs,
    _attempt_record,
    _bootstrap,
    _decisions,
    _economic_evidence_record,
    _family_record,
    _int_field,
    _interval_record,
    _list_field,
    _market_schema_record,
    _membership_hash,
    _reason_codes,
    _select_floor,
    _select_samples,
    _signals_and_reasons_at_floor,
    _split_validation,
    _string_list_field,
    _validate_disjoint_manifest_memberships,
    _write_immutable,
    verify_research_inputs,
)
from trading_bot.research_run import load_research_bars
from trading_bot.uncertainty import ResidualIntervalCalibrator, fit_residual_interval
from trading_bot.view_features import (
    VIEW_FIELDS,
    ViewSample,
    build_view_samples,
    c0_view_feature_schema,
    fit_view_matrix,
    transform_view_matrix,
)

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised without the neural group
    raise NeuralDependencyError(INSTALL_HINT) from error

_REPORT_VERSION = "1.0.0"
_STATUS = "development_only"
_LOGISTIC_L2 = Decimal("0.1")
_LOGISTIC_ITERATIONS = 100
_LOGISTIC_LEARNING_RATE = Decimal("0.05")
_RESIDUAL_COVERAGE = Decimal("0.80")
_BOOTSTRAP_CONFIDENCE = Decimal("0.95")
_LATENCY_REPETITIONS = 20
_VRAM_HEADROOM = Decimal("0.75")
_CODE_DEPENDENCY_FILES = (
    "abstention.py",
    "evaluation.py",
    "feature_matrix.py",
    "hypothesis_registry.py",
    "logistic_baseline.py",
    "market_features.py",
    "phase_a_fold_run.py",
    "uncertainty.py",
    "view_features.py",
    "neural/_torch.py",
    "neural/compute_evidence.py",
    "neural/fold_run.py",
    "neural/mixture.py",
    "neural/spec.py",
    "neural/training.py",
    "neural/windows.py",
)


class NeuralFoldError(PhaseAFoldError):
    """A fail-closed neural fold failure carrying the Phase-A error-code contract."""


@dataclass(frozen=True, slots=True)
class NeuralFoldConfig:
    capture_root: Path
    split_manifest_path: Path
    audit_path: Path
    registry_path: Path
    output_path: Path
    checkpoint_path: Path
    hypothesis_family_id: UUID
    attempt_id: UUID
    spec: CtmMixtureSpec
    encoder_kind: EncoderKind
    gate_kind: GateKind
    seed_ordinal: Literal[0, 1, 2]
    fold_index: int
    device: Literal["cpu", "cuda"] = "cpu"
    block_length: int = 96
    bootstrap_repetitions: int = 2000
    minimum_selection_trades: int = 200

    def __post_init__(self) -> None:
        if self.encoder_kind == "ctm":
            raise ValueError("ENCODER_NOT_AVAILABLE: the CTM encoder arrives with plan C.1")
        if self.gate_kind == "learned" and self.encoder_kind == "none":
            raise ValueError("learned gate requires an encoder")
        if self.gate_kind != "learned" and self.encoder_kind != "none":
            raise ValueError("encoder-free gate kinds require encoder_kind 'none'")
        if self.fold_index not in (0, 1, 2):
            raise ValueError("fold index must be zero, one, or two")
        for value in (self.block_length, self.bootstrap_repetitions, self.minimum_selection_trades):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("neural fold numeric settings must be positive integers")

    @property
    def random_seed(self) -> int:
        return self.spec.seeds[self.seed_ordinal]


@dataclass(frozen=True, slots=True)
class NeuralFoldArtifact:
    output_path: Path
    checkpoint_path: Path
    report_hash: str
    checkpoint_hash: str
    attempt_id: UUID
    horizon_bars: Literal[4, 16]
    encoder_kind: EncoderKind
    gate_kind: GateKind
    seed_ordinal: Literal[0, 1, 2]
    fold_index: Literal[0, 1, 2]


@dataclass(frozen=True, slots=True)
class _Partition:
    samples: tuple[EnrichedBarSample, ...]
    positions: MembershipPositions


@dataclass(frozen=True, slots=True)
class _Base:
    model: LogisticVectorModel
    calibrator: PlattCalibrator
    mean_positive: Decimal
    mean_nonpositive: Decimal
    logits: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class _Forecast:
    probabilities: tuple[Decimal, ...]
    certainties: tuple[Decimal, ...]
    gate_weights: tuple[tuple[Decimal, ...], ...]
    expert_residuals: tuple[tuple[Decimal, ...], ...]
    encoder_residuals: tuple[Decimal | None, ...]
    base_logits: tuple[Decimal, ...]
    points: tuple[Decimal, ...]


def run_neural_fold(config: NeuralFoldConfig) -> NeuralFoldArtifact:
    """Verify inputs, train one cell, evaluate its locked test fold, publish immutably."""
    research = verify_research_inputs(
        capture_root=config.capture_root,
        split_manifest_path=config.split_manifest_path,
        audit_path=config.audit_path,
        fold_index=config.fold_index,
    )
    _verify_campaign(config, research)
    try:
        return _execute(config, research)
    except PhaseAFoldError as error:
        _record(config, research.horizon_bars, status="failed", artifact_hash=None,
                error_code=error.error_code)
        raise NeuralFoldError(str(error), error_code=error.error_code) from error


def _verify_campaign(config: NeuralFoldConfig, research: ResearchInputs) -> None:
    with HypothesisRegistry(config.registry_path) as registry:
        family = registry.get_family(config.hypothesis_family_id)
        attempt = registry.get_attempt(config.attempt_id)
    if family is None:
        _fail("hypothesis family is not registered", "FAMILY_NOT_REGISTERED")
    if family.name != NEURAL_FAMILY_NAME:
        _fail("hypothesis family is not the neural family", "FAMILY_NOT_NEURAL")
    if attempt is None:
        _fail("hypothesis attempt is not reserved", "ATTEMPT_NOT_RESERVED")
    if attempt.family_id != family.family_id:
        _fail("hypothesis attempt family does not match", "ATTEMPT_FAMILY_MISMATCH")
    if family.audit_report_hash != research.audit.report_hash:
        _fail("hypothesis audit report hash does not match", "AUDIT_HASH_MISMATCH")
    if family.feature_schema_hash != research.schema.schema_hash:
        _fail("hypothesis feature schema hash does not match", "FEATURE_SCHEMA_HASH_MISMATCH")
    reserved = (
        family.h4_split_manifest_hash if research.horizon_bars == 4 else family.h16_split_manifest_hash
    )
    if reserved != research.split_hash:
        _fail("hypothesis split manifest hash does not match", "SPLIT_MANIFEST_HASH_MISMATCH")
    expected = neural_campaign_config_hash(
        family,
        config.spec,
        block_length=config.block_length,
        bootstrap_repetitions=config.bootstrap_repetitions,
        minimum_selection_trades=config.minimum_selection_trades,
    )
    if attempt.config_hash != expected:
        _fail("hypothesis attempt config hash does not match", "ATTEMPT_CONFIG_HASH_MISMATCH")
    if config.spec.feature_schema_hash != research.schema.schema_hash:
        _fail("spec feature schema hash does not match", "SPEC_SCHEMA_HASH_MISMATCH")
    if config.spec.view_schema_hash != c0_view_feature_schema().schema_hash:
        _fail("spec view schema hash does not match", "VIEW_SCHEMA_HASH_MISMATCH")


def _execute(config: NeuralFoldConfig, research: ResearchInputs) -> NeuralFoldArtifact:
    if config.output_path.exists():
        _fail("neural fold output already exists and is immutable", "OUTPUT_ALREADY_EXISTS")
    if config.checkpoint_path.exists():
        _fail("neural fold checkpoint already exists and is immutable", "CHECKPOINT_ALREADY_EXISTS")
    seed_everything(config.random_seed)
    spec = config.spec
    window = spec.window_bars
    test_end_ns = _int_field(research.fold, "test_end_ns")
    bars = load_research_bars(config.capture_root / "dataset", available_before_ns=test_end_ns)
    primary = tuple(bar for bar in bars if bar.venue == "OKX")
    reference = tuple(bar for bar in bars if bar.venue == "BINANCE")
    enriched = build_enriched_samples(
        primary, reference, horizon_bars=research.horizon_bars, schema=research.schema
    )
    views = build_view_samples(enriched, primary, schema=c0_view_feature_schema())
    views_by_id = {view.base.base.sample_id: view for view in views}
    samples_by_id = {sample.base.sample_id: sample for sample in enriched}

    train_ids = tuple(_string_list_field(research.fold, "train_ids"))
    validation_ids = tuple(_string_list_field(research.fold, "validation_ids"))
    test_ids = tuple(_string_list_field(research.fold, "test_ids"))
    _validate_disjoint_manifest_memberships(train_ids, validation_ids, test_ids)
    training = _select_samples(samples_by_id, train_ids)
    validation = _select_samples(samples_by_id, validation_ids)
    testing = _select_samples(samples_by_id, test_ids)
    calibration, selection = _split_validation(validation)
    calibration_ids = tuple(sample.base.sample_id for sample in calibration)
    selection_ids = tuple(sample.base.sample_id for sample in selection)
    _validate_disjoint_manifest_memberships(train_ids, calibration_ids, selection_ids, test_ids)

    market_matrix = fit_feature_matrix(training, fields=MARKET_FIELD_ORDER, schema=research.schema)
    view_matrix = fit_view_matrix(
        tuple(views_by_id[sample.base.sample_id] for sample in training), fields=VIEW_FIELDS
    )
    layout = ColumnLayout(MARKET_FIELD_ORDER, VIEW_FIELDS)
    rows = build_row_tensor(
        transform_feature_matrix(market_matrix, enriched),
        transform_view_matrix(view_matrix, views),
        decision_times_ns=tuple(sample.decision_time_ns for sample in enriched),
        layout=layout,
    )
    base = _fit_base(market_matrix, training, calibration, enriched)

    partitions = {
        name: _Partition(samples, membership_positions(rows, ids, window=window))
        for name, samples, ids in (
            ("train", training, train_ids),
            ("calibration", calibration, calibration_ids),
            ("selection", selection, selection_ids),
            ("test", testing, test_ids),
        )
    }
    if partitions["test"].positions.dropped_ids:
        _fail("a test sample lacks a full trailing window", "TEST_WINDOW_UNAVAILABLE")
    for name in ("train", "calibration", "selection"):
        if partitions[name].positions.positions.numel() == 0:
            _fail(f"{name} partition has no windowed samples", "MEMBERSHIP_INVALID")

    labels = torch.tensor(
        [1.0 if sample.base.forward_return > 0 else 0.0 for sample in enriched], dtype=torch.float32
    )
    data = TrainingData(
        rows=rows,
        base_logits=torch.tensor([float(value) for value in base.logits], dtype=torch.float32),
        labels=labels,
        train_positions=partitions["train"].positions.positions,
        calibration_positions=partitions["calibration"].positions.positions,
        window=window,
    )
    _vram_preflight(config, layout, data)
    model, training_record, best_single = _fit_mixture(config, layout, data)

    forecasts = {
        name: _forecast(model, data, base, partitions[name])
        for name in ("calibration", "selection", "test")
    }
    calibration_kept = _kept(calibration, partitions["calibration"])
    selection_kept = _kept(selection, partitions["selection"])
    residual_interval = fit_residual_interval(
        forecasts["calibration"].points,
        tuple(sample.base.forward_return for sample in calibration_kept),
        membership_ids=tuple(sample.base.sample_id for sample in calibration_kept),
        decision_times_ns=tuple(sample.decision_time_ns for sample in calibration_kept),
        coverage=_RESIDUAL_COVERAGE,
    )
    selection_decisions = _decisions(
        forecasts["selection"].points, residual_interval,
        research.audit.adverse_costs, research.audit.adverse_spread_bps,
    )
    threshold, floor, candidate_floors, selected_signals, selected_result = _select_threshold(
        selection_decisions, forecasts["selection"].certainties, selection_kept,
        grid=spec.certainty_threshold_grid, minimum_trades=config.minimum_selection_trades,
        costs=research.audit.adverse_costs, spread_bps=research.audit.adverse_spread_bps,
    )
    masked_selection = _mask(selection_decisions, forecasts["selection"].certainties, threshold)
    _, selection_reasons = _signals_and_reasons_at_floor(masked_selection, floor)

    test_decisions = _decisions(
        forecasts["test"].points, residual_interval,
        research.audit.adverse_costs, research.audit.adverse_spread_bps,
    )
    masked_test = _mask(test_decisions, forecasts["test"].certainties, threshold)
    test_signals, test_reasons = _signals_and_reasons_at_floor(masked_test, floor)
    outcomes = tuple(sample.base.forward_return for sample in testing)
    base_result = evaluate_signals(
        test_signals, outcomes, (research.audit.base_spread_bps,) * len(testing),
        research.audit.base_costs,
    )
    adverse_result = evaluate_signals(
        test_signals, outcomes, (research.audit.adverse_spread_bps,) * len(testing),
        research.audit.adverse_costs,
    )
    active_base = tuple(
        net for signal, net in zip(test_signals, base_result.net_returns, strict=True) if signal != 0
    )
    effective_block = min(config.block_length, len(active_base)) if active_base else 0
    bootstrap = (
        _bootstrap(active_base, block_length=effective_block,
                   repetitions=config.bootstrap_repetitions, seed=config.random_seed)
        if active_base
        else None
    )
    raw_p_value = Decimal(1) if bootstrap is None else bootstrap["one_sided_p_value"]
    fold_count = len(_list_field(research.split, "folds"))
    reasons = _reason_codes(
        fold_count=fold_count, base=base_result, adverse=adverse_result, bootstrap=bootstrap
    )

    checkpoint_hash = _save_checkpoint(model, config.checkpoint_path)
    latency_p50, latency_p95 = _latency(model, data, partitions["test"])
    compute = ComputeEvidence(
        parameter_count=count_parameters(model),
        peak_vram_bytes=peak_vram_bytes(),
        training_seconds=training_record.training_seconds,
        inference_p50_ms=latency_p50,
        inference_p95_ms=latency_p95,
        gpu_seconds=gpu_seconds(training_record.training_seconds, device=config.device),
    )
    code_manifest = _code_dependency_manifest()
    code_hash = content_sha256(code_manifest)
    with HypothesisRegistry(config.registry_path) as registry:
        family = registry.get_family(config.hypothesis_family_id)
        attempt = registry.get_attempt(config.attempt_id)
    if family is None or attempt is None:
        _fail("hypothesis registration vanished during the run", "ATTEMPT_NOT_RESERVED")

    material: dict[str, object] = {
        "report_version": _REPORT_VERSION,
        "status": _STATUS,
        "reason_codes": list(reasons),
        "capture_root_hash": research.capture_hash,
        "dataset_root_hash": research.dataset_hash,
        "audit_report_hash": research.audit.report_hash,
        "cost_capture_root_hash": research.audit.cost_capture_root_hash,
        "economic_evidence": _economic_evidence_record(research.audit),
        "split_manifest_hash": research.split_hash,
        "feature_schema_hash": research.schema.schema_hash,
        "feature_matrix_schema_hash": market_matrix.schema_hash,
        "view_schema_hash": c0_view_feature_schema().schema_hash,
        "view_matrix_schema_hash": view_matrix.schema_hash,
        "market_feature_schema": _market_schema_record(research.schema),
        "column_layout": layout.record(),
        "hypothesis_family_id": str(family.family_id),
        "hypothesis_name": family.name,
        "hypothesis_family": _family_record(family),
        "attempt_id": str(attempt.attempt_id),
        "attempt_config_hash": attempt.config_hash,
        "hypothesis_attempt": _attempt_record(attempt),
        "spec": {**spec.canonical_record(), "spec_hash": spec.spec_hash},
        "code_dependency_manifest": code_manifest,
        "code_hash": code_hash,
        "horizon_bars": research.horizon_bars,
        "encoder_kind": config.encoder_kind,
        "gate_kind": config.gate_kind,
        "seed_ordinal": config.seed_ordinal,
        "random_seed": config.random_seed,
        "device": config.device,
        "fold_index": config.fold_index,
        "fold_count": fold_count,
        "test_end_ns": test_end_ns,
        "train_sample_count": len(training),
        "validation_sample_count": len(validation),
        "calibration_sample_count": len(calibration_kept),
        "selection_sample_count": len(selection_kept),
        "test_sample_count": len(testing),
        "train_membership_ids": list(train_ids),
        "window_unavailable_train_ids": list(partitions["train"].positions.dropped_ids),
        "window_unavailable_calibration_ids": list(partitions["calibration"].positions.dropped_ids),
        "window_unavailable_selection_ids": list(partitions["selection"].positions.dropped_ids),
        "calibration_membership_ids": [s.base.sample_id for s in calibration_kept],
        "selection_membership_ids": [s.base.sample_id for s in selection_kept],
        "test_membership_ids": list(test_ids),
        "train_membership_hash": _membership_hash(training),
        "calibration_membership_hash": _membership_hash(calibration_kept),
        "selection_membership_hash": _membership_hash(selection_kept),
        "test_membership_hash": _membership_hash(testing),
        "train_decision_times_ns": [s.decision_time_ns for s in training],
        "calibration_decision_times_ns": [s.decision_time_ns for s in calibration_kept],
        "selection_decision_times_ns": [s.decision_time_ns for s in selection_kept],
        "test_decision_times_ns": [s.decision_time_ns for s in testing],
        "block_length": config.block_length,
        "effective_block_length": effective_block,
        "bootstrap_repetitions": config.bootstrap_repetitions,
        "bootstrap_confidence": _BOOTSTRAP_CONFIDENCE,
        "base_model": _base_record(base),
        "training": training_record.record(),
        "best_single_selection": best_single,
        "checkpoint_relative_path": config.checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "compute_evidence": compute.record(),
        "residual_interval": _interval_record(residual_interval),
        "selection_threshold": {
            "kind": "certainty_then_conservative_ev_floor",
            "certainty_threshold_grid": list(spec.certainty_threshold_grid),
            "chosen_certainty_threshold": threshold,
            "candidate_floors": list(candidate_floors),
            "chosen_floor": floor,
            "minimum_trade_count": config.minimum_selection_trades,
            "selected_trade_count": sum(signal != 0 for signal in selected_signals),
            "selected_adverse_total_net_return": selected_result.total_net_return,
        },
        **_forecast_records("calibration", forecasts["calibration"]),
        **_forecast_records("selection", forecasts["selection"]),
        "selection_forward_returns": [s.base.forward_return for s in selection_kept],
        "selection_original_signals": [d.signal for d in selection_decisions],
        "selection_conservative_expected_values": [
            d.adverse_expected_value for d in selection_decisions
        ],
        "selection_abstention_reason_codes": list(selection_reasons),
        "selection_signals": list(selected_signals),
        **_forecast_records("test", forecasts["test"]),
        "test_forward_returns": list(outcomes),
        "test_original_signals": [d.signal for d in test_decisions],
        "test_conservative_expected_values": [d.adverse_expected_value for d in test_decisions],
        "test_abstention_reason_codes": list(test_reasons),
        "test_signals": list(test_signals),
        "base_evaluation": _evaluation_record(base_result, research.audit.base_spread_bps),
        "adverse_evaluation": _evaluation_record(adverse_result, research.audit.adverse_spread_bps),
        "active_base_net_returns": list(active_base),
        "base_active_median_net_return": Decimal(median(active_base)) if active_base else Decimal(0),
        "base_bootstrap": bootstrap,
        "raw_one_sided_p_value": raw_p_value,
        "fold_local_q_value": raw_p_value,
        "q_value_scope": "provisional_fold_local_single_candidate",
        "final_holdout_status": "locked",
        "final_holdout_read_count": 0,
    }
    report_hash = content_sha256(material)
    document = dict(material)
    document["report_hash"] = report_hash
    _write_immutable(config.output_path, document)
    _record(config, research.horizon_bars, status="completed", artifact_hash=report_hash,
            error_code=None)
    return NeuralFoldArtifact(
        output_path=config.output_path,
        checkpoint_path=config.checkpoint_path,
        report_hash=report_hash,
        checkpoint_hash=checkpoint_hash,
        attempt_id=attempt.attempt_id,
        horizon_bars=research.horizon_bars,
        encoder_kind=config.encoder_kind,
        gate_kind=config.gate_kind,
        seed_ordinal=config.seed_ordinal,
        fold_index=_fold_literal(config.fold_index),
    )


def _fit_base(
    market_matrix: FittedFeatureMatrix,
    training: tuple[EnrichedBarSample, ...],
    calibration: tuple[EnrichedBarSample, ...],
    everything: tuple[EnrichedBarSample, ...],
) -> _Base:
    train_rows = transform_feature_matrix(market_matrix, training).rows
    targets = tuple(sample.base.forward_return for sample in training)
    positive = tuple(target for target in targets if target > 0)
    nonpositive = tuple(target for target in targets if target <= 0)
    if not positive or not nonpositive:
        _fail("logistic training requires both direction classes", "LOGISTIC_CLASS_MISSING")
    model = fit_logistic_rows(
        train_rows, tuple(1 if target > 0 else 0 for target in targets),
        l2=_LOGISTIC_L2, iterations=_LOGISTIC_ITERATIONS, learning_rate=_LOGISTIC_LEARNING_RATE,
    )
    calibrator = fit_platt_calibrator_rows(
        model,
        transform_feature_matrix(market_matrix, calibration).rows,
        tuple(1 if sample.base.forward_return > 0 else 0 for sample in calibration),
        l2=_LOGISTIC_L2, iterations=_LOGISTIC_ITERATIONS, learning_rate=_LOGISTIC_LEARNING_RATE,
    )
    all_rows = transform_feature_matrix(market_matrix, everything).rows
    logits = tuple(
        calibrator.slope * _row_logit(model, row) + calibrator.intercept for row in all_rows
    )
    return _Base(
        model=model,
        calibrator=calibrator,
        mean_positive=sum(positive, Decimal(0)) / Decimal(len(positive)),
        mean_nonpositive=sum(nonpositive, Decimal(0)) / Decimal(len(nonpositive)),
        logits=logits,
    )


def _row_logit(model: LogisticVectorModel, row: tuple[Decimal, ...]) -> Decimal:
    value = model.intercept + sum(
        (coefficient * feature for coefficient, feature in zip(model.coefficients, row, strict=True)),
        Decimal(0),
    )
    if not value.is_finite():
        _fail("base logistic logit is not finite", "BASE_LOGIT_NOT_FINITE")
    return value


def _vram_preflight(config: NeuralFoldConfig, layout: ColumnLayout, data: TrainingData) -> None:
    if config.device != "cuda":
        return
    if not torch.cuda.is_available():
        _fail("cuda device requested but unavailable", "VRAM_PREFLIGHT_FAILED")
    torch.cuda.reset_peak_memory_stats()
    probe = ResidualMixture(
        layout, expert_hidden=config.spec.expert_hidden, gate_kind="learned",
        encoder=GruEncoder(layout.width, config.spec.gru_hidden, len(EXPERT_ORDER)),
    ).to("cuda")
    batch = data.train_positions[: config.spec.batch_size]
    windows = gather_windows(data.rows, batch, window=data.window).to("cuda")
    output = probe(data.rows.values[batch].to("cuda"), windows, data.base_logits[batch].to("cuda"))
    output.logit.sum().backward()
    peak = Decimal(peak_vram_bytes())
    del probe, windows, output
    torch.cuda.empty_cache()
    if peak > Decimal(config.spec.vram_limit_bytes) * _VRAM_HEADROOM:
        _fail("mixture does not fit the VRAM limit with headroom", "VRAM_PREFLIGHT_FAILED")


def _fit_mixture(
    config: NeuralFoldConfig, layout: ColumnLayout, data: TrainingData
) -> tuple[ResidualMixture, TrainingRecord, dict[str, object] | None]:
    spec = config.spec
    common = {
        "batch_size": spec.batch_size, "learning_rate": spec.learning_rate,
        "max_epochs": spec.max_epochs, "patience": spec.patience,
        "load_balance_weight": spec.load_balance_weight, "seed": config.random_seed,
        "device": config.device,
    }
    if config.gate_kind == "none":
        model = ResidualMixture(layout, expert_hidden=spec.expert_hidden, gate_kind="none", encoder=None)
        model.to(config.device).eval()
        return model, TrainingRecord(0, 0, Decimal(str(evaluate_loss(
            model, data, data.calibration_positions, batch_size=spec.batch_size,
            load_balance_weight=float(spec.load_balance_weight)))), False, Decimal(0)), None
    if config.gate_kind == "best_single":
        candidates: list[tuple[float, int, ResidualMixture, TrainingRecord]] = []
        for index, _ in enumerate(EXPERT_ORDER):
            candidate = ResidualMixture(
                layout, expert_hidden=spec.expert_hidden, gate_kind="best_single",
                encoder=None, selected_expert=index,
            )
            record = train_mixture(candidate, data, **common)
            candidates.append((float(record.best_calibration_loss), index, candidate, record))
        loss, index, model, record = min(candidates, key=lambda item: (item[0], item[1]))
        return model, record, {
            "selected_expert": EXPERT_ORDER[index],
            "calibration_losses": {
                EXPERT_ORDER[i]: Decimal(str(c[0])) for i, c in enumerate(candidates)
            },
        }
    encoder = (
        GruEncoder(layout.width, spec.gru_hidden, len(EXPERT_ORDER))
        if config.gate_kind == "learned"
        else None
    )
    model = ResidualMixture(
        layout, expert_hidden=spec.expert_hidden, gate_kind=config.gate_kind, encoder=encoder
    )
    record = train_mixture(model, data, **common)
    return model, record, None


def _forecast(
    model: ResidualMixture, data: TrainingData, base: _Base, partition: _Partition
) -> _Forecast:
    device = next(model.parameters()).device
    positions = partition.positions.positions
    model.eval()
    with torch.no_grad():
        current = data.rows.values[positions].to(device)
        windows = (
            gather_windows(data.rows, positions, window=data.window).to(device)
            if model.encoder is not None
            else None
        )
        output = model(current, windows, data.base_logits[positions].to(device))
        probabilities = output.probability.cpu()
        certainties = binary_certainty(probabilities)
        weights = output.gate_weights.cpu()
        residuals = output.expert_residuals.cpu()
        encoder_residuals = (
            None if output.encoder_residual is None else output.encoder_residual.cpu()
        )
    probability_values = tuple(Decimal(str(value)) for value in probabilities.tolist())
    encoder_values: tuple[Decimal | None, ...] = (
        (None,) * len(probability_values)
        if encoder_residuals is None
        else tuple(Decimal(str(value)) for value in encoder_residuals.tolist())
    )
    return _Forecast(
        probabilities=probability_values,
        certainties=tuple(Decimal(str(value)) for value in certainties.tolist()),
        gate_weights=tuple(tuple(Decimal(str(v)) for v in row) for row in weights.tolist()),
        expert_residuals=tuple(tuple(Decimal(str(v)) for v in row) for row in residuals.tolist()),
        encoder_residuals=encoder_values,
        base_logits=tuple(base.logits[int(position)] for position in positions.tolist()),
        points=tuple(
            probability * base.mean_positive + (Decimal(1) - probability) * base.mean_nonpositive
            for probability in probability_values
        ),
    )


def _kept(
    samples: tuple[EnrichedBarSample, ...], partition: _Partition
) -> tuple[EnrichedBarSample, ...]:
    kept = set(partition.positions.sample_ids)
    return tuple(sample for sample in samples if sample.base.sample_id in kept)


def _mask(
    decisions: tuple[AbstentionDecision, ...],
    certainties: tuple[Decimal, ...],
    threshold: Decimal,
) -> tuple[AbstentionDecision, ...]:
    return tuple(
        AbstentionDecision(0, Decimal(0), "CERTAINTY_BELOW_THRESHOLD")
        if decision.signal != 0 and certainty < threshold
        else decision
        for decision, certainty in zip(decisions, certainties, strict=True)
    )


def _select_threshold(
    decisions: tuple[AbstentionDecision, ...],
    certainties: tuple[Decimal, ...],
    samples: tuple[EnrichedBarSample, ...],
    *,
    grid: tuple[Decimal, ...],
    minimum_trades: int,
    costs: CostScenario,
    spread_bps: Decimal,
) -> tuple[Decimal, Decimal, tuple[Decimal, ...], tuple[int, ...], EvaluationResult]:
    scored: list[
        tuple[Decimal, Decimal, Decimal, tuple[Decimal, ...], tuple[int, ...], EvaluationResult]
    ] = []
    for threshold in grid:
        masked = _mask(decisions, certainties, threshold)
        try:
            floor, candidate_floors, signals, result = _select_floor(
                masked, samples, minimum_trades=minimum_trades, costs=costs, spread_bps=spread_bps
            )
        except PhaseAFoldError as error:
            if error.error_code != "INSUFFICIENT_SELECTION_TRADES":
                raise
            continue
        scored.append((result.total_net_return, threshold, floor, candidate_floors, signals, result))
    if not scored:
        _fail("insufficient selected validation trades", "INSUFFICIENT_SELECTION_TRADES")
    _, threshold, floor, candidate_floors, signals, result = max(
        scored, key=lambda item: (item[0], item[1], item[2])
    )
    return threshold, floor, candidate_floors, signals, result


def _latency(
    model: ResidualMixture, data: TrainingData, partition: _Partition
) -> tuple[Decimal, Decimal]:
    encoder: GruEncoder | None = None
    if model.encoder is not None:
        if not isinstance(model.encoder, GruEncoder):
            _fail("latency replay supports only the GRU encoder in C.0", "ENCODER_NOT_AVAILABLE")
        encoder = GruEncoder(model.layout.width, model.encoder.gru.hidden_size, len(EXPERT_ORDER))
    cpu_model = ResidualMixture(
        model.layout,
        expert_hidden=model.expert_hidden,
        gate_kind=model.gate_kind,
        encoder=encoder,
        selected_expert=model.selected_expert,
    )
    cpu_model.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})
    cpu_model.eval()
    position = partition.positions.positions[:1]
    current = data.rows.values[position]
    windows = (
        gather_windows(data.rows, position, window=data.window) if cpu_model.encoder is not None else None
    )
    base = data.base_logits[position]

    def forward() -> None:
        with torch.no_grad():
            cpu_model(current, windows, base)

    return measure_inference_latency(forward, repetitions=_LATENCY_REPETITIONS)


def _save_checkpoint(model: ResidualMixture, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    config: NeuralFoldConfig,
    horizon_bars: Literal[4, 16],
    *,
    status: Literal["completed", "failed"],
    artifact_hash: str | None,
    error_code: str | None,
) -> None:
    try:
        with HypothesisRegistry(config.registry_path) as registry:
            registry.record_neural_fold_run(
                NeuralFoldRun(
                    attempt_id=config.attempt_id,
                    horizon_bars=horizon_bars,
                    encoder_kind=config.encoder_kind,
                    gate_kind=config.gate_kind,
                    seed_ordinal=config.seed_ordinal,
                    fold_index=_fold_literal(config.fold_index),
                    status=status,
                    artifact_hash=artifact_hash,
                    error_code=error_code,
                    recorded_at_ns=time_ns(),
                )
            )
    except RegistryConflictError:
        if status == "completed":
            raise
        # A previously recorded terminal outcome stays authoritative and immutable.


def _base_record(base: _Base) -> dict[str, object]:
    return {
        "kind": "calibrated_logistic_all_market_fields",
        "intercept": base.model.intercept,
        "coefficients": list(base.model.coefficients),
        "platt_slope": base.calibrator.slope,
        "platt_intercept": base.calibrator.intercept,
        "mean_positive_training_return": base.mean_positive,
        "mean_nonpositive_training_return": base.mean_nonpositive,
        "l2": _LOGISTIC_L2,
        "iterations": _LOGISTIC_ITERATIONS,
        "learning_rate": _LOGISTIC_LEARNING_RATE,
    }


def _forecast_records(prefix: str, forecast: _Forecast) -> dict[str, object]:
    return {
        f"{prefix}_probabilities": list(forecast.probabilities),
        f"{prefix}_certainties": list(forecast.certainties),
        f"{prefix}_gate_weights": [list(row) for row in forecast.gate_weights],
        f"{prefix}_expert_residuals": [list(row) for row in forecast.expert_residuals],
        f"{prefix}_encoder_residuals": list(forecast.encoder_residuals),
        f"{prefix}_base_logits": list(forecast.base_logits),
        f"{prefix}_point_forecasts": list(forecast.points),
    }


def _evaluation_record(result: EvaluationResult, spread_bps: Decimal) -> dict[str, object]:
    return {
        "scenario_name": result.scenario_name,
        "observed_spread_bps": spread_bps,
        "trade_count": result.trade_count,
        "total_net_return": result.total_net_return,
        "mean_net_return": result.mean_net_return,
        "win_rate": result.win_rate,
        "maximum_drawdown": result.maximum_drawdown,
        "net_returns": list(result.net_returns),
    }


def _code_dependency_manifest() -> list[dict[str, str]]:
    root = Path(__file__).parent.parent
    return [
        {
            "relative_path": name,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
        for name in _CODE_DEPENDENCY_FILES
    ]


def _fold_literal(value: int) -> Literal[0, 1, 2]:
    if value == 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    raise ValueError("fold index must be zero, one, or two")


def _fail(message: str, error_code: str) -> NoReturn:
    raise NeuralFoldError(message, error_code=error_code)
```

Complete the import block with `from typing import Literal, NoReturn`, `from trading_bot.feature_matrix import FittedFeatureMatrix, fit_feature_matrix, transform_feature_matrix`, and `from trading_bot.strategy import CostScenario`.

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/neural/test_fold_run.py -q -p no:cacheprovider
```

Expected: `4 passed`. The first test takes roughly one to two minutes: the view builder computes the 1,344-bar autocorrelation in Decimal for ≈ 2,000 fixture samples and the base logistic runs 100 pure-Python gradient iterations.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/fold_run.py tests/neural/test_fold_run.py
git commit -m "feat: run registered neural mixture folds"
```

### Task 12: Fold verifier and encoder-free comparators

**Files:**
- Modify: `src/trading_bot/neural/fold_run.py`
- Modify: `tests/neural/test_fold_run.py`

**Interfaces:**
- Produces:

```python
def verify_neural_fold(path: Path) -> bool
def verify_neural_fold_document(value: object, *, checkpoint_root: Path | None) -> bool
```

  A document passes only when: `report_hash` equals the canonical hash of the material; every linkage field listed below is a 64-hex string; `final_holdout_status == "locked"` and `final_holdout_read_count == 0`; the `train`/`calibration`/`selection`/`test` membership hashes recompute from the stored IDs and decision times; `base_evaluation` and `adverse_evaluation` totals and trade counts recompute from `test_signals`, `test_forward_returns`, the stored spread, and the stored cost scenario in `economic_evidence`; `active_base_net_returns` recompute from the base evaluation; `base_bootstrap` recomputes from `active_base_net_returns`, `random_seed`, `effective_block_length`, `bootstrap_repetitions`; `reason_codes` recompute; `test_gate_weights` rows sum to one within `1e-5`; `encoder_kind`/`gate_kind`/`seed_ordinal` are valid literals and `random_seed == spec.seeds[seed_ordinal]`; `spec.spec_hash` recomputes from the spec record; and, when `checkpoint_root` is given, the checkpoint file's SHA-256 equals `checkpoint_sha256`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/neural/test_fold_run.py`:

```python
from trading_bot.canonical import canonical_json, content_sha256
from trading_bot.neural.fold_run import verify_neural_fold


def _rewrite(path: Path, mutate: "Callable[[dict[str, object]], None]") -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    document.pop("report_hash")
    document["report_hash"] = content_sha256(document)
    path.write_bytes(canonical_json(document))


def test_verifier_accepts_published_fold_and_rejects_tampering(tmp_path: Path) -> None:
    config = neural_fold_fixture(tmp_path)
    run_neural_fold(config)

    assert verify_neural_fold(config.output_path)

    def flip_first_signal(document: dict[str, object]) -> None:
        signals = document["test_signals"]
        assert isinstance(signals, list)
        signals[0] = 1 if signals[0] != 1 else -1

    _rewrite(config.output_path, flip_first_signal)
    assert not verify_neural_fold(config.output_path)

    run_neural_fold(replace(config, output_path=tmp_path / "second.json",
                            checkpoint_path=tmp_path / "second.pt", seed_ordinal=1))
    _rewrite(tmp_path / "second.json", lambda d: d.update(final_holdout_read_count=1))
    assert not verify_neural_fold(tmp_path / "second.json")

    run_neural_fold(replace(config, output_path=tmp_path / "third.json",
                            checkpoint_path=tmp_path / "third.pt", seed_ordinal=2))
    (tmp_path / "third.pt").write_bytes(b"tampered")
    assert not verify_neural_fold(tmp_path / "third.json")


@pytest.mark.parametrize(
    "encoder_kind, gate_kind, expected_weights",
    (
        ("none", "none", [0, 0, 0, 0]),
        ("none", "uniform", [Decimal("0.25")] * 4),
        ("none", "best_single", None),
    ),
)
def test_encoder_free_comparators_publish_fixed_gate_weights(
    tmp_path: Path, encoder_kind: EncoderKind, gate_kind: GateKind,
    expected_weights: list[Decimal] | None,
) -> None:
    config = neural_fold_fixture(tmp_path, encoder_kind=encoder_kind, gate_kind=gate_kind)

    artifact = run_neural_fold(config)
    document = json.loads(config.output_path.read_text(encoding="utf-8"))

    assert verify_neural_fold(config.output_path)
    assert artifact.gate_kind == gate_kind
    weights = [Decimal(value) for value in document["test_gate_weights"][0]]
    if expected_weights is None:
        assert weights.count(Decimal(1)) == 1 and weights.count(Decimal(0)) == 3
        assert document["best_single_selection"]["selected_expert"] in (
            "slope", "spectral", "rate", "summary"
        )
    else:
        assert weights == expected_weights
    assert document["test_encoder_residuals"][0] is None
    if gate_kind == "none":
        assert document["training"]["epochs_run"] == 0
        for probability, logit in zip(
            document["test_probabilities"], document["test_base_logits"], strict=True
        ):
            sigmoid = Decimal(1) / (Decimal(1) + (-Decimal(logit)).exp())
            assert abs(Decimal(probability) - sigmoid) < Decimal("1e-5")
```

Add `from collections.abc import Callable` to the test imports and drop the string quotes on the annotation.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/neural/test_fold_run.py -q -p no:cacheprovider -k "verifier or comparators"
```

Expected: FAIL with `ImportError: cannot import name 'verify_neural_fold'`.

- [ ] **Step 3: Add the verifier**

Append to `neural/fold_run.py`:

```python
_LINKAGE_HASH_FIELDS = (
    "capture_root_hash",
    "dataset_root_hash",
    "audit_report_hash",
    "cost_capture_root_hash",
    "split_manifest_hash",
    "feature_schema_hash",
    "feature_matrix_schema_hash",
    "view_schema_hash",
    "view_matrix_schema_hash",
    "attempt_config_hash",
    "code_hash",
    "checkpoint_sha256",
)
_GATE_WEIGHT_TOLERANCE = Decimal("1e-5")


def verify_neural_fold(path: Path) -> bool:
    """Strongly verify one published neural fold report and its checkpoint file."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"), parse_float=_reject_float)
    except (OSError, ValueError):
        return False
    return verify_neural_fold_document(document, checkpoint_root=path.parent)


def verify_neural_fold_document(value: object, *, checkpoint_root: Path | None) -> bool:
    if not isinstance(value, dict):
        return False
    document: dict[str, object] = dict(value)
    try:
        reported_hash = document.pop("report_hash")
        if reported_hash != content_sha256(document):
            return False
        if document.get("report_version") != _REPORT_VERSION or document.get("status") != _STATUS:
            return False
        if document.get("final_holdout_status") != "locked":
            return False
        if document.get("final_holdout_read_count") != 0:
            return False
        for field in _LINKAGE_HASH_FIELDS:
            _sha256(document, field)
        spec = CtmMixtureSpec.model_validate(
            {key: item for key, item in _object(document, "spec").items() if key != "spec_hash"}
        )
        if _object(document, "spec").get("spec_hash") != spec.spec_hash:
            return False
        seed_ordinal = _int(document, "seed_ordinal")
        if seed_ordinal not in (0, 1, 2) or _int(document, "random_seed") != spec.seeds[seed_ordinal]:
            return False
        if document.get("encoder_kind") not in ("none", "gru", "ctm"):
            return False
        if document.get("gate_kind") not in ("none", "uniform", "best_single", "learned"):
            return False
        for prefix, samples_field in (
            ("train", "train_membership_ids"),
            ("calibration", "calibration_membership_ids"),
            ("selection", "selection_membership_ids"),
            ("test", "test_membership_ids"),
        ):
            ids = _strings(document, samples_field)
            times = _ints(document, f"{prefix}_decision_times_ns")
            expected = content_sha256([[i, t] for i, t in zip(ids, times, strict=True)])
            if document.get(f"{prefix}_membership_hash") != expected:
                return False
        signals = _signals(document, "test_signals")
        outcomes = _decimals(document, "test_forward_returns")
        evidence = _object(document, "economic_evidence")
        for role, evaluation_key in (("base", "base_evaluation"), ("adverse", "adverse_evaluation")):
            role_record = _object(evidence, role)
            scenario = _scenario(_object(role_record, "scenario"))
            spread = _decimal(role_record, "observed_spread_bps")
            evaluation = _object(document, evaluation_key)
            replay = evaluate_signals(signals, outcomes, (spread,) * len(signals), scenario)
            if (
                _decimal(evaluation, "total_net_return") != replay.total_net_return
                or _int(evaluation, "trade_count") != replay.trade_count
                or _decimals(evaluation, "net_returns") != replay.net_returns
            ):
                return False
        base_evaluation = _object(document, "base_evaluation")
        active = tuple(
            net for signal, net in zip(signals, _decimals(base_evaluation, "net_returns"), strict=True)
            if signal != 0
        )
        if _decimals(document, "active_base_net_returns") != active:
            return False
        effective = _int(document, "effective_block_length")
        expected_bootstrap = (
            _bootstrap(active, block_length=effective,
                       repetitions=_int(document, "bootstrap_repetitions"),
                       seed=_int(document, "random_seed"))
            if active
            else None
        )
        stored_bootstrap = document.get("base_bootstrap")
        if expected_bootstrap is None:
            if stored_bootstrap is not None:
                return False
        else:
            if not isinstance(stored_bootstrap, dict):
                return False
            if {k: _decimal(stored_bootstrap, k) for k in expected_bootstrap} != expected_bootstrap:
                return False
        base_record = _object(evidence, "base")
        adverse_record = _object(evidence, "adverse")
        base_result = evaluate_signals(
            signals, outcomes,
            (_decimal(base_record, "observed_spread_bps"),) * len(signals),
            _scenario(_object(base_record, "scenario")),
        )
        adverse_result = evaluate_signals(
            signals, outcomes,
            (_decimal(adverse_record, "observed_spread_bps"),) * len(signals),
            _scenario(_object(adverse_record, "scenario")),
        )
        expected_reasons = _reason_codes(
            fold_count=_int(document, "fold_count"), base=base_result, adverse=adverse_result,
            bootstrap=expected_bootstrap,
        )
        if tuple(_strings(document, "reason_codes")) != expected_reasons:
            return False
        for row in _list(document, "test_gate_weights"):
            if not isinstance(row, list):
                return False
            total = sum((Decimal(str(item)) for item in row), Decimal(0))
            if document.get("gate_kind") != "none" and abs(total - Decimal(1)) > _GATE_WEIGHT_TOLERANCE:
                return False
        if checkpoint_root is not None:
            checkpoint = checkpoint_root / _string(document, "checkpoint_relative_path")
            if not checkpoint.is_file():
                return False
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != document["checkpoint_sha256"]:
                return False
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False
    return True


def _scenario(record: dict[str, object]) -> CostScenario:
    return CostScenario(
        _string(record, "name"),
        _decimal(record, "fee_bps_per_side"),
        _decimal(record, "spread_multiplier"),
        _decimal(record, "slippage_bps_per_side"),
        _decimal(record, "funding_bps"),
    )


def _reject_float(text: str) -> object:
    raise ValueError(f"fold documents must not contain binary floats: {text}")


def _object(record: Mapping[str, object], key: str) -> dict[str, object]:
    value = record[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return {str(k): v for k, v in value.items()}


def _list(record: Mapping[str, object], key: str) -> list[object]:
    value = record[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _string(record: Mapping[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _sha256(record: Mapping[str, object], key: str) -> str:
    value = _string(record, key)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{key} must be a lowercase sha256 hex digest")
    return value


def _int(record: Mapping[str, object], key: str) -> int:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _decimal(record: Mapping[str, object], key: str) -> Decimal:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"{key} must be a canonical decimal string or integer")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"{key} must be finite")
    return result


def _decimals(record: Mapping[str, object], key: str) -> tuple[Decimal, ...]:
    return tuple(
        Decimal(item) if isinstance(item, str | int) and not isinstance(item, bool) else _raise(key)
        for item in _list(record, key)
    )


def _strings(record: Mapping[str, object], key: str) -> list[str]:
    return [item if isinstance(item, str) else _raise(key) for item in _list(record, key)]


def _ints(record: Mapping[str, object], key: str) -> list[int]:
    return [
        item if isinstance(item, int) and not isinstance(item, bool) else _raise(key)
        for item in _list(record, key)
    ]


def _signals(record: Mapping[str, object], key: str) -> tuple[int, ...]:
    return tuple(item if item in (-1, 0, 1) else _raise(key) for item in _ints(record, key))


def _raise(key: str) -> NoReturn:
    raise TypeError(f"{key} contains an invalid element")
```

Add `import json` to the module imports (`CostScenario` is already imported since Task 11). The `economic_evidence` block written by the reused Phase-A `_economic_evidence_record` has the shape `{"audit_report_hash", "cost_capture_root_hash", "audited_costs", "base": {"spread_role", "observed_spread_bps", "scenario": {"name", "fee_bps_per_side", "spread_multiplier", "slippage_bps_per_side", "funding_bps"}}, "adverse": {...same...}}`; the verifier above reads exactly those keys.

- [ ] **Step 4: Run the tests**

```powershell
uv run pytest tests/neural/test_fold_run.py -q -p no:cacheprovider
```

Expected: `8 passed`.

- [ ] **Step 5: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/neural/fold_run.py tests/neural/test_fold_run.py
git commit -m "feat: verify neural fold reports and add comparator gates"
```

---

### Task 13: CLI commands and README section

**Files:**
- Modify: `src/trading_bot/cli.py` (parsers after `phase-a-fold`, dispatch after the `phase-a-fold` branch)
- Modify: `README.md`
- Create: `tests/neural/test_cli.py`

**Interfaces:**
- Produces two subcommands:
  - `neural-register-campaign --workspace-root --spec --audit --registry --h4-split-manifest --h16-split-manifest --family-id [--revision-reason] [--block-length 96] [--bootstrap-repetitions 2000] [--minimum-selection-trades 200]` → prints `{"attempt_id": ..., "attempt_index": ..., "config_hash": ...}` as canonical JSON.
  - `neural-fold --workspace-root --capture --split-manifest --audit --registry --spec --output --checkpoint --family-id --attempt-id --encoder-kind {none,gru} --gate-kind {none,uniform,best_single,learned} --seed-ordinal {0,1,2} --fold-index [--device {cpu,cuda}] [--block-length 96] [--bootstrap-repetitions 2000] [--minimum-selection-trades 200]` → prints `{"output": ..., "report_hash": ..., "checkpoint_sha256": ...}`.
- Both import `trading_bot.neural.*` lazily inside the dispatch branch so the CLI stays usable without torch for every other command; `neural-register-campaign` needs no torch at all.

- [ ] **Step 1: Write the failing test**

`tests/neural/test_cli.py`:

```python
import json
from pathlib import Path

from tests.neural.test_fold_run import FAMILY_ID, neural_fold_fixture, test_spec
from trading_bot.cli import main


def test_cli_registers_campaign_and_runs_fold(tmp_path: Path, capsys: "pytest.CaptureFixture[str]") -> None:
    fixture = neural_fold_fixture(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(test_spec().canonical_record()), encoding="utf-8")
    registry_path = tmp_path / "cli-neural.sqlite3"

    assert main([
        "neural-register-campaign",
        "--workspace-root", str(tmp_path),
        "--spec", str(spec_path),
        "--audit", str(fixture.audit_path),
        "--registry", str(registry_path),
        "--h4-split-manifest", str(fixture.split_manifest_path),
        "--h16-split-manifest", str(fixture.split_manifest_path),
        "--family-id", str(FAMILY_ID),
        "--block-length", "1",
        "--bootstrap-repetitions", "20",
        "--minimum-selection-trades", "1",
    ]) == 0
    registration = json.loads(capsys.readouterr().out)
    assert registration["attempt_index"] == 0

    assert main([
        "neural-fold",
        "--workspace-root", str(tmp_path),
        "--capture", str(fixture.capture_root),
        "--split-manifest", str(fixture.split_manifest_path),
        "--audit", str(fixture.audit_path),
        "--registry", str(registry_path),
        "--spec", str(spec_path),
        "--output", str(tmp_path / "cli-fold.json"),
        "--checkpoint", str(tmp_path / "cli-fold.pt"),
        "--family-id", str(FAMILY_ID),
        "--attempt-id", registration["attempt_id"],
        "--encoder-kind", "none",
        "--gate-kind", "uniform",
        "--seed-ordinal", "0",
        "--fold-index", "0",
        "--block-length", "1",
        "--bootstrap-repetitions", "20",
        "--minimum-selection-trades", "1",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (tmp_path / "cli-fold.json").exists()
    assert len(result["report_hash"]) == 64
```

Add `import pytest` and drop the annotation quotes. The `--h16-split-manifest` reuses the h4 manifest in this test; the family binds whatever hash the file has, and the fold only checks the h4 hash.

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/neural/test_cli.py -q -p no:cacheprovider
```

Expected: FAIL with an argparse error (`invalid choice: 'neural-register-campaign'`), exit code 2 raised as `SystemExit`.

- [ ] **Step 3: Add the parsers and dispatch**

In `cli.py`, after the `phase-a-fold` parser block:

```python
    neural_register = commands.add_parser("neural-register-campaign")
    neural_register.add_argument("--workspace-root", type=Path, default=Path.cwd())
    neural_register.add_argument("--spec", type=Path, required=True)
    neural_register.add_argument("--audit", type=Path, required=True)
    neural_register.add_argument("--registry", type=Path, required=True)
    neural_register.add_argument("--h4-split-manifest", type=Path, required=True)
    neural_register.add_argument("--h16-split-manifest", type=Path, required=True)
    neural_register.add_argument("--family-id", type=UUID, required=True)
    neural_register.add_argument("--revision-reason", default=None)
    neural_register.add_argument("--block-length", type=int, default=96)
    neural_register.add_argument("--bootstrap-repetitions", type=int, default=2000)
    neural_register.add_argument("--minimum-selection-trades", type=int, default=200)
    neural_fold = commands.add_parser("neural-fold")
    neural_fold.add_argument("--workspace-root", type=Path, default=Path.cwd())
    for name in ("--capture", "--split-manifest", "--audit", "--registry", "--spec",
                 "--output", "--checkpoint"):
        neural_fold.add_argument(name, type=Path, required=True)
    neural_fold.add_argument("--family-id", type=UUID, required=True)
    neural_fold.add_argument("--attempt-id", type=UUID, required=True)
    neural_fold.add_argument("--encoder-kind", choices=("none", "gru"), required=True)
    neural_fold.add_argument(
        "--gate-kind", choices=("none", "uniform", "best_single", "learned"), required=True
    )
    neural_fold.add_argument("--seed-ordinal", type=int, choices=(0, 1, 2), required=True)
    neural_fold.add_argument("--fold-index", type=int, required=True)
    neural_fold.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    neural_fold.add_argument("--block-length", type=int, default=96)
    neural_fold.add_argument("--bootstrap-repetitions", type=int, default=2000)
    neural_fold.add_argument("--minimum-selection-trades", type=int, default=200)
```

Dispatch, after the `phase-a-fold` branch:

```python
    if parsed.command == "neural-register-campaign":
        from trading_bot.neural.spec import load_mixture_spec, neural_family, register_neural_campaign
        from trading_bot.phase_a_audit import verify_phase_a_audit
        from trading_bot.walk_forward_run import verify_walk_forward_manifest

        workspace = parsed.workspace_root.resolve()
        spec_path, audit_path, registry_path, h4_path, h16_path = (
            _bounded_workspace_path(workspace, path, label="neural campaign")
            for path in (parsed.spec, parsed.audit, parsed.registry,
                         parsed.h4_split_manifest, parsed.h16_split_manifest)
        )
        if not verify_phase_a_audit(audit_path):
            raise ValueError("neural campaign audit failed verification")
        for manifest in (h4_path, h16_path):
            if not verify_walk_forward_manifest(manifest):
                raise ValueError("neural campaign split manifest failed verification")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        h4 = json.loads(h4_path.read_text(encoding="utf-8"))
        h16 = json.loads(h16_path.read_text(encoding="utf-8"))
        spec = load_mixture_spec(spec_path)
        family = neural_family(
            family_id=parsed.family_id,
            feature_schema_hash=spec.feature_schema_hash,
            audit_report_hash=str(audit["report_hash"]),
            h4_split_manifest_hash=str(h4["manifest_hash"]),
            h16_split_manifest_hash=str(h16["manifest_hash"]),
        )
        attempt = register_neural_campaign(
            registry_path, family, spec,
            block_length=parsed.block_length,
            bootstrap_repetitions=parsed.bootstrap_repetitions,
            minimum_selection_trades=parsed.minimum_selection_trades,
            revision_reason=parsed.revision_reason,
        )
        sys.stdout.write(canonical_json({
            "attempt_id": str(attempt.attempt_id),
            "attempt_index": attempt.attempt_index,
            "config_hash": attempt.config_hash,
        }).decode("utf-8") + "\n")
        return 0
    if parsed.command == "neural-fold":
        from trading_bot.neural.fold_run import NeuralFoldConfig, run_neural_fold
        from trading_bot.neural.spec import load_mixture_spec

        workspace = parsed.workspace_root.resolve()
        paths = tuple(
            _bounded_workspace_path(workspace, path, label="neural fold")
            for path in (parsed.capture, parsed.split_manifest, parsed.audit, parsed.registry,
                         parsed.spec, parsed.output, parsed.checkpoint)
        )
        artifact = run_neural_fold(
            NeuralFoldConfig(
                capture_root=paths[0],
                split_manifest_path=paths[1],
                audit_path=paths[2],
                registry_path=paths[3],
                output_path=paths[5],
                checkpoint_path=paths[6],
                hypothesis_family_id=parsed.family_id,
                attempt_id=parsed.attempt_id,
                spec=load_mixture_spec(paths[4]),
                encoder_kind=parsed.encoder_kind,
                gate_kind=parsed.gate_kind,
                seed_ordinal=parsed.seed_ordinal,
                fold_index=parsed.fold_index,
                device=parsed.device,
                block_length=parsed.block_length,
                bootstrap_repetitions=parsed.bootstrap_repetitions,
                minimum_selection_trades=parsed.minimum_selection_trades,
            )
        )
        sys.stdout.write(canonical_json({
            "output": str(artifact.output_path),
            "report_hash": artifact.report_hash,
            "checkpoint_sha256": artifact.checkpoint_hash,
        }).decode("utf-8") + "\n")
        return 0
```

Match the module's existing output convention (check how `phase-a-fold` prints; if it uses `print(json.dumps(...))`, use the same helper). `parsed.seed_ordinal` is an `int`; narrow it with the same pattern the module uses for `fold_index`, or pass through `_fold_literal`-style narrowing in `NeuralFoldConfig` (its `seed_ordinal` annotation is `Literal[0, 1, 2]`; argparse `choices` enforces the value at runtime, and mypy needs a cast or a small `_seed_ordinal_literal(value: int) -> Literal[0, 1, 2]` helper next to the dispatch).

- [ ] **Step 4: Add the README section**

Insert before `## Harte Grenzen`:

````markdown
## Neuronale Residual-Mixture (C.0, research_only)

Die C.0-Stufe der CTM-Spec (`docs/superpowers/specs/2026-09-08-ctm-moe-challenger-design.md`)
besteht aus kausalen Steigungs- und Spektral-Views, vier flachen Residual-Experten und
einem GRU-Encoder, der die Gate-Gewichte erzeugt. Alles bleibt residual über der
eingefrorenen Phase-A-Logistic. PyTorch ist eine optionale Abhängigkeit:

```powershell
uv sync --group neural
```

Die Kampagne wird in einer **eigenen** Registry-Datei registriert (niemals in der
Phase-A-Registry) und läuft pro Zelle aus Horizont, Encoder, Gate, Seed und Fold:

```powershell
uv run trading-research neural-register-campaign `
  --workspace-root . `
  --spec configs/neural-mixture-v1.json `
  --audit artifacts/phase-a-audit-btc-15m-580d.json `
  --registry artifacts/neural/hypotheses-neural.sqlite3 `
  --h4-split-manifest artifacts/walk-forward-btc-15m-h4-580d.json `
  --h16-split-manifest artifacts/walk-forward-btc-15m-h16-580d.json `
  --family-id 00000000-0000-0000-0000-000000000901

uv run trading-research neural-fold `
  --workspace-root . `
  --capture data/captures/2026-08-24-btc-15m-580d `
  --split-manifest artifacts/walk-forward-btc-15m-h4-580d.json `
  --audit artifacts/phase-a-audit-btc-15m-580d.json `
  --registry artifacts/neural/hypotheses-neural.sqlite3 `
  --spec configs/neural-mixture-v1.json `
  --output artifacts/neural/gru-moe-h4-fold0-s0.json `
  --checkpoint artifacts/neural/gru-moe-h4-fold0-s0.pt `
  --family-id 00000000-0000-0000-0000-000000000901 `
  --attempt-id <attempt-id> `
  --encoder-kind gru --gate-kind learned --seed-ordinal 0 --fold-index 0 `
  --device cuda
```

Der Fold-Report bindet Capture, Audit, Split, Feature- und View-Schema, Spec-Hash,
Checkpoint-Hash und Compute-Evidenz (Parameter, Peak-VRAM, Trainingszeit,
Inferenzlatenz). Test-Samples ohne vollständiges 96-Bar-Fenster brechen den Lauf ab;
der finale Holdout wird nie gelesen. Ein Ergebnis dieser Stufe hat keine
Promotionsautorität; der CTM-Encoder und die Entscheidung folgen in Plan C.1/C.2.
````

- [ ] **Step 5: Run the CLI test and the whole suite**

```powershell
uv run pytest tests/neural/test_cli.py -q -p no:cacheprovider
uv run pytest -q -p no:cacheprovider
```

Expected: CLI test passes; full suite green (≈ 511 + new tests).

- [ ] **Step 6: Lint, typecheck, commit**

```powershell
uv run ruff check src tests; uv run ruff format --check src tests; uv run mypy src tests
git add src/trading_bot/cli.py README.md tests/neural/test_cli.py
git commit -m "feat: expose neural campaign registration and fold commands"
```

---

### Task 14: Real-data C.0 checkpoint (h4, fold 0, `gru_moe`, seed 0) — research_only

**Files:**
- Create: `configs/neural-mixture-v1.json`
- Create: `.superpowers/sdd/2026-09-08-c0-residual-mixture/task-14-report.md`

This task produces no economic decision and registers no `eligible` claim. It proves the pipeline runs end-to-end on the frozen 580-day capture and records protocol-10 compute evidence.

- [ ] **Step 1: Locate the frozen inputs and confirm the audit exists**

```powershell
Get-ChildItem artifacts | Where-Object Name -match 'walk-forward.*h(4|16).*580d|phase-a-audit'
```

Expected: one h4 manifest, one h16 manifest, one Phase-A audit JSON for `data/captures/2026-08-24-btc-15m-580d`. **If no audit exists**, stop and ask the user: building it requires `capture-public-cost-evidence` (network access to OKX/Binance public endpoints) followed by `phase-a-audit`; that is an outward-facing action and needs explicit approval.

- [ ] **Step 2: Write the frozen spec file**

`configs/neural-mixture-v1.json` (Decimals as strings; obtain the two schema hashes with `uv run python -c "from trading_bot.market_features import phase_a_market_feature_schema as m; from trading_bot.view_features import c0_view_feature_schema as v; print(m().schema_hash); print(v().schema_hash)"`):

```json
{
  "version": "1.0.0",
  "reference_commit": "<40-hex SHA of the reviewed SakanaAI/continuous-thought-machines commit>",
  "reference_source_hash": null,
  "feature_schema_hash": "<market schema hash>",
  "view_schema_hash": "<view schema hash>",
  "window_bars": 96,
  "d_model": 256,
  "d_input": 64,
  "heads": 4,
  "iterations": 20,
  "memory_length": 16,
  "n_synch_out": 64,
  "n_synch_action": 32,
  "gru_hidden": 64,
  "expert_hidden": 8,
  "experts": ["slope", "spectral", "rate", "summary"],
  "load_balance_weight": "0.01",
  "certainty_threshold_grid": ["0", "0.02", "0.05", "0.10", "0.15", "0.20", "0.30"],
  "seeds": [20260908, 20260909, 20260910],
  "batch_size": 256,
  "learning_rate": "0.001",
  "max_epochs": 30,
  "patience": 5,
  "vram_limit_bytes": 12884901888
}
```

The `reference_commit` is the tip of `main` in the Sakana repository at review time (`git ls-remote https://github.com/SakanaAI/continuous-thought-machines main`); record the date it was read in the report.

- [ ] **Step 3: Register the campaign**

```powershell
uv run trading-research neural-register-campaign --workspace-root . --spec configs/neural-mixture-v1.json --audit <audit.json> --registry artifacts/neural/hypotheses-neural.sqlite3 --h4-split-manifest <h4.json> --h16-split-manifest <h16.json> --family-id 00000000-0000-0000-0000-000000000901
```

Expected: JSON with `attempt_index: 0`. Save `attempt_id`.

- [ ] **Step 4: Run the four C.0 configurations for h4 fold 0, seed 0**

```powershell
foreach ($cell in @(@("none","none"), @("none","uniform"), @("none","best_single"), @("gru","learned"))) {
  $name = "$($cell[0])-$($cell[1])-h4-fold0-s0"
  uv run trading-research neural-fold --workspace-root . --capture data/captures/2026-08-24-btc-15m-580d --split-manifest <h4.json> --audit <audit.json> --registry artifacts/neural/hypotheses-neural.sqlite3 --spec configs/neural-mixture-v1.json --output artifacts/neural/$name.json --checkpoint artifacts/neural/$name.pt --family-id 00000000-0000-0000-0000-000000000901 --attempt-id <attempt-id> --encoder-kind $cell[0] --gate-kind $cell[1] --seed-ordinal 0 --fold-index 0 --device cuda
}
```

Expected: four reports; `gru-learned` training under 60 minutes on the RTX 3060. A `VRAM_PREFLIGHT_FAILED` or a 2-hour timeout is a registered failure, not a reason to change the spec.

- [ ] **Step 5: Verify and record**

```powershell
uv run python -c "from pathlib import Path; from trading_bot.neural.fold_run import verify_neural_fold; print([verify_neural_fold(p) for p in Path('artifacts/neural').glob('*-h4-fold0-s0.json')])"
```

Expected: `[True, True, True, True]`. Write the report with: attempt id, four report hashes, `compute_evidence` per cell (parameter count, peak VRAM, training seconds, inference p50/p95), `window_unavailable_train_ids` count, chosen certainty threshold and floor, base/adverse totals and trade counts, and the sentence "No decision artifact was produced; C.0 outputs are research_only." Commit `configs/` and the report (artifacts stay gitignored).

```powershell
git add configs/neural-mixture-v1.json .superpowers/sdd/2026-09-08-c0-residual-mixture/task-14-report.md
git commit -m "docs: record c0 real-data checkpoint"
```

---

## Final C.0 Review Gate

C.0 is complete when all of the following hold, verified by running the commands, not by assertion:

- `uv run pytest -q -p no:cacheprovider` passes with TEMP on `C:`; `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run mypy src tests` are clean with the `neural` group installed.
- `uv run python -c "import trading_bot.neural.spec"` succeeds in an environment **without** torch (`uv sync` with no `--group neural` in a throwaway checkout), and `import trading_bot.neural.windows` there raises `NeuralDependencyError` mentioning `uv sync --group neural`.
- The four h4 fold-0 reports from Task 14 exist, verify, and carry `final_holdout_read_count: 0`, `status: development_only`, and non-empty `compute_evidence`.
- The neural registry file contains exactly one family (`ctm_regime_gated_mixture`), one attempt, and four completed `neural_fold_runs`; the Phase-A registry is untouched (`git status` shows no change under `artifacts/` beyond `artifacts/neural/`, which is gitignored).
- The spec's section 6.2 and 6.5 edits from Task 4 Step 5 are committed.
- No file under `src/trading_bot/` outside `neural/` imports torch.

Plan 2 (`C.1/C.2`: vendored CTM reference at the pinned commit, `CtmEncoder` returning `EncoderOutput` with a per-tick `certainty_trace`, dual-tick loss in `mixture_loss`, `ctm_alone` and `ctm_moe` cells, three seeds × three folds × two horizons, and the immutable section-7 decision artifact) is written only after this gate passes.
