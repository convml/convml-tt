from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("convml_tt")
except PackageNotFoundError:
    __version__ = "unknown version"
