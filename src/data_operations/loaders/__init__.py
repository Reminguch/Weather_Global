"""Dataset loader utilities."""

__all__ = ["open_graphcast_era5"]


def __getattr__(name: str):
    if name == "open_graphcast_era5":
        from .graphcast_dataset import open_graphcast_era5

        return open_graphcast_era5
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
