import os as _os

_legacy_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "critics"))
if _legacy_path not in __path__:
    __path__.append(_legacy_path)

