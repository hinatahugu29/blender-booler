"""Booler: Topology-preserving SubD-ready Boolean add-on for Blender."""

from __future__ import annotations

from .ui import operators, panel, properties


def register() -> None:
    properties.register()
    operators.register()
    panel.register()


def unregister() -> None:
    panel.unregister()
    operators.unregister()
    properties.unregister()
