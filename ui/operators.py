"""Blender Operators for Booler (creation, non-destructive rebuild, and apply/bake)."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from ..core.pipeline import BoolerConfig, run_booler_pipeline, update_booler_object
from ..geometry.metrics import evaluate_mesh


class BOOLER_OT_execute(bpy.types.Operator):
    """Execute topology-preserving SubD-ready Boolean on two selected meshes"""

    bl_idname = "booler.execute"
    bl_label = "Booler Boolean"
    bl_options = {"REGISTER", "UNDO"}

    operation: EnumProperty(
        name="Operation",
        items=[
            ("UNION", "Union", "Combine two meshes"),
            ("DIFFERENCE", "Difference", "Subtract selected from active"),
            ("INTERSECT", "Intersect", "Keep overlapping volume"),
        ],
        default="UNION",
    )
    solver: EnumProperty(
        name="Solver",
        items=[
            ("EXACT", "Exact", "Exact arithmetic solver"),
            ("MANIFOLD", "Manifold", "Fast manifold solver"),
            ("FLOAT", "Float", "Fast floating-point solver"),
        ],
        default="EXACT",
    )
    collar_width: FloatProperty(name="Collar Width", default=0.05, min=0.002, max=1.0)
    collar_rings: IntProperty(name="Collar Rings", default=1, min=1, max=4)
    fillet_segments: IntProperty(
        name="Fillet Segments",
        description="0 = sharp support crease, >=1 = smooth C1 quad fillet",
        default=0,
        min=0,
        max=6,
    )
    fillet_profile: FloatProperty(name="Fillet Profile", default=0.5, min=0.1, max=1.0)
    enable_subd_fit: BoolProperty(
        name="SubD Limit Fit",
        description="Inverse-fit local control vertices to preserve Catmull-Clark limit surface",
        default=True,
    )
    non_destructive: BoolProperty(
        name="Non-Destructive",
        description="Hide source objects and link them for live parameter editing",
        default=True,
    )
    add_subd_preview: BoolProperty(name="Add SubD Preview", default=True)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        selected_meshes = [o for o in context.selected_objects if o.type == "MESH"]
        return (
            context.active_object is not None
            and context.active_object.type == "MESH"
            and len(selected_meshes) == 2
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        obj_a = context.active_object
        others = [o for o in context.selected_objects if o.type == "MESH" and o != obj_a]
        if not others:
            self.report({"ERROR"}, "Select exactly two mesh objects.")
            return {"CANCELLED"}
        obj_b = others[0]

        cfg = BoolerConfig(
            operation=self.operation,
            solver=self.solver,
            collar_width=self.collar_width,
            collar_rings=self.collar_rings,
            fillet_segments=self.fillet_segments,
            fillet_profile=self.fillet_profile,
            enable_subd_fit=self.enable_subd_fit,
        )
        result_obj, _ = run_booler_pipeline(
            obj_a=obj_a,
            obj_b=obj_b,
            config=cfg,
            result_name=f"{obj_a.name}_Booler",
        )

        if self.add_subd_preview:
            subd = result_obj.modifiers.new("Subdivision", "SUBSURF")
            subd.levels = 2
            subd.render_levels = 2

        report = evaluate_mesh(result_obj, name=result_obj.name)

        if self.non_destructive:
            bs = result_obj.booler
            bs.auto_update = False
            bs.is_booler_result = True
            bs.source_a = obj_a
            bs.source_b = obj_b
            bs.operation = self.operation
            bs.solver = self.solver
            bs.collar_width = self.collar_width
            bs.collar_rings = self.collar_rings
            bs.fillet_segments = self.fillet_segments
            bs.fillet_profile = self.fillet_profile
            bs.enable_subd_fit = self.enable_subd_fit
            bs.last_summary = (
                f"Q:{report.quad_count} T:{report.tri_count} N:{report.ngon_count} | "
                f"NM:{report.non_manifold_edges} | Poles:{report.intersection_irregular_poles}"
            )
            bs.auto_update = True

            for src in (obj_a, obj_b):
                src.hide_set(True)
                src.hide_viewport = True
                src.hide_render = True

        # Select the new result object
        bpy.ops.object.select_all(action="DESELECT")
        result_obj.select_set(True)
        context.view_layer.objects.active = result_obj

        self.report({"INFO"}, report.summary_line())
        return {"FINISHED"}


class BOOLER_OT_rebuild(bpy.types.Operator):
    """Rebuild the active non-destructive Booler result from its linked source objects"""

    bl_idname = "booler.rebuild"
    bl_label = "Rebuild Booler"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and hasattr(obj, "booler")
            and obj.booler.is_booler_result
            and obj.booler.source_a is not None
            and obj.booler.source_b is not None
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        obj = context.active_object
        bs = obj.booler
        update_booler_object(obj, bs.source_a, bs.source_b, bs.to_config())
        report = evaluate_mesh(obj, name=obj.name)
        bs.last_summary = (
            f"Q:{report.quad_count} T:{report.tri_count} N:{report.ngon_count} | "
            f"NM:{report.non_manifold_edges} | Poles:{report.intersection_irregular_poles}"
        )
        self.report({"INFO"}, report.summary_line())
        return {"FINISHED"}


class BOOLER_OT_apply(bpy.types.Operator):
    """Bake active Booler result into a regular mesh and unlink sources"""

    bl_idname = "booler.apply"
    bl_label = "Apply / Bake Booler"
    bl_options = {"REGISTER", "UNDO"}

    delete_sources: BoolProperty(
        name="Delete Source Objects",
        default=False,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and hasattr(obj, "booler") and obj.booler.is_booler_result

    def execute(self, context: bpy.types.Context) -> set[str]:
        obj = context.active_object
        bs = obj.booler
        src_a, src_b = bs.source_a, bs.source_b
        bs.auto_update = False
        bs.is_booler_result = False
        bs.source_a = None
        bs.source_b = None
        bs.last_summary = ""

        if self.delete_sources:
            for src in (src_a, src_b):
                if src is not None and src.name in bpy.data.objects:
                    bpy.data.objects.remove(src, do_unlink=True)
        return {"FINISHED"}


CLASSES = (BOOLER_OT_execute, BOOLER_OT_rebuild, BOOLER_OT_apply)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
