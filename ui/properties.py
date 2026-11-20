"""Non-destructive PropertyGroup attached to Booler result objects."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from ..core.pipeline import BoolerConfig, update_booler_object
from ..geometry.metrics import evaluate_mesh

_IS_UPDATING = False


def _on_booler_param_update(self: "BoolerObjectSettings", context: bpy.types.Context) -> None:
    """In-place rebuild callback when any non-destructive parameter changes."""
    global _IS_UPDATING
    if _IS_UPDATING or not self.is_booler_result or not self.auto_update:
        return
    obj = getattr(self, "id_data", None)
    if obj is None or obj.type != "MESH":
        return
    if self.source_a is None or self.source_b is None:
        return
    if self.source_a.type != "MESH" or self.source_b.type != "MESH":
        return

    _IS_UPDATING = True
    try:
        cfg = self.to_config()
        update_booler_object(obj, self.source_a, self.source_b, cfg)
        report = evaluate_mesh(obj, name=obj.name)
        self.last_summary = (
            f"Q:{report.quad_count} T:{report.tri_count} N:{report.ngon_count} | "
            f"NM:{report.non_manifold_edges} | Poles:{report.intersection_irregular_poles}"
        )
    except Exception as exc:
        self.last_summary = f"Error: {exc}"
    finally:
        _IS_UPDATING = False


def _on_toggle_wire_sources(self: "BoolerObjectSettings", context: bpy.types.Context) -> None:
    """Toggle viewport visibility and wireframe display of linked source objects."""
    for src in (self.source_a, self.source_b):
        if src is None:
            continue
        if self.show_sources_wire:
            src.hide_set(False)
            src.hide_viewport = False
            src.display_type = "WIRE"
        else:
            src.hide_set(True)
            src.hide_viewport = True


class BoolerObjectSettings(bpy.types.PropertyGroup):
    is_booler_result: BoolProperty(name="Is Booler Result", default=False)
    auto_update: BoolProperty(name="Live Update", default=True)
    show_sources_wire: BoolProperty(
        name="Show Sources (Wire)",
        default=False,
        update=_on_toggle_wire_sources,
    )
    source_a: PointerProperty(name="Target (A)", type=bpy.types.Object)
    source_b: PointerProperty(name="Cutter (B)", type=bpy.types.Object)

    operation: EnumProperty(
        name="Operation",
        items=[
            ("UNION", "Union", "Combine two meshes"),
            ("DIFFERENCE", "Difference", "Subtract Cutter from Target"),
            ("INTERSECT", "Intersect", "Keep overlapping volume"),
        ],
        default="UNION",
        update=_on_booler_param_update,
    )
    solver: EnumProperty(
        name="Solver",
        items=[
            ("EXACT", "Exact", "Exact arithmetic solver"),
            ("MANIFOLD", "Manifold", "Fast manifold solver"),
            ("FLOAT", "Float", "Fast floating-point solver"),
        ],
        default="EXACT",
        update=_on_booler_param_update,
    )
    collar_width: FloatProperty(
        name="Collar Width",
        default=0.05,
        min=0.002,
        max=1.0,
        step=1,
        precision=3,
        update=_on_booler_param_update,
    )
    collar_rings: IntProperty(
        name="Collar Rings",
        default=1,
        min=1,
        max=4,
        update=_on_booler_param_update,
    )
    fillet_segments: IntProperty(
        name="Fillet Segments",
        description="0 = sharp support crease, >=1 = smooth C1 quad fillet",
        default=0,
        min=0,
        max=6,
        update=_on_booler_param_update,
    )
    fillet_profile: FloatProperty(
        name="Fillet Profile",
        default=0.5,
        min=0.1,
        max=1.0,
        update=_on_booler_param_update,
    )
    enable_subd_fit: BoolProperty(
        name="SubD Limit Fit",
        default=True,
        update=_on_booler_param_update,
    )
    last_summary: StringProperty(name="Quality Summary", default="")

    def to_config(self) -> BoolerConfig:
        return BoolerConfig(
            operation=self.operation,
            solver=self.solver,
            collar_width=self.collar_width,
            collar_rings=self.collar_rings,
            fillet_segments=self.fillet_segments,
            fillet_profile=self.fillet_profile,
            enable_subd_fit=self.enable_subd_fit,
        )


def register() -> None:
    bpy.utils.register_class(BoolerObjectSettings)
    bpy.types.Object.booler = PointerProperty(type=BoolerObjectSettings)


def unregister() -> None:
    if hasattr(bpy.types.Object, "booler"):
        del bpy.types.Object.booler
    bpy.utils.unregister_class(BoolerObjectSettings)
