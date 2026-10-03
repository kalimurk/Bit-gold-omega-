"""OBTC chain package: multi-timeframe FullChain assembly.

Re-exports are lazy (PEP 562) so that ``obtc.chain.models`` can be
imported without pulling in the builder (which depends on the detection
package, which itself imports the models). Any import order works.
"""
from __future__ import annotations

_LAZY = {
    "ChainBuilder": "obtc.chain.builder",
    "Bias": "obtc.chain.models",
    "BlockState": "obtc.chain.models",
    "ChainEvent": "obtc.chain.models",
    "ChainLink": "obtc.chain.models",
    "ChainState": "obtc.chain.models",
    "Direction": "obtc.chain.models",
    "FairValueGap": "obtc.chain.models",
    "FullChain": "obtc.chain.models",
    "MarketShift": "obtc.chain.models",
    "OrderBlock": "obtc.chain.models",
    "Sweep": "obtc.chain.models",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
