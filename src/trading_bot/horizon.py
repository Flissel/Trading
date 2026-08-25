"""Prediction-horizon parsing shared by immutable research artifacts."""


def manifest_horizon_bars(manifest: dict[str, object]) -> int:
    """Return a validated horizon, preserving compatibility with h1 manifests."""
    value = manifest.get("horizon_bars", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("manifest horizon_bars must be a positive integer")
    return value
