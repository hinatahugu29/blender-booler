"""N-Panel UI for Booler."""

from __future__ import annotations

import bpy


class BOOLER_PT_main(bpy.types.Panel):
    bl_label = "Booler"
    bl_idname = "BOOLER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Booler"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        # Section 1: Create new Booler Boolean
        box = layout.box()
        box.label(text="Create (Select 2 Meshes)", icon="MOD_BOOLEAN")
        row = box.row(align=True)
        op_u = row.operator("booler.execute", text="Union")
        op_u.operation = "UNION"
        op_d = row.operator("booler.execute", text="Diff")
        op_d.operation = "DIFFERENCE"
        op_i = row.operator("booler.execute", text="Intersect")
        op_i.operation = "INTERSECT"

        # Section 2: Non-destructive controls for active Booler result
        obj = context.active_object
        if obj is not None and obj.type == "MESH" and hasattr(obj, "booler") and obj.booler.is_booler_result:
            bs = obj.booler
            edit_box = layout.box()
            edit_box.label(text="Live Booler Settings", icon="MOD_SUBSURF")
            edit_box.prop(bs, "operation")
            edit_box.prop(bs, "solver")
            edit_box.prop(bs, "collar_width")
            edit_box.prop(bs, "collar_rings")
            edit_box.prop(bs, "fillet_segments")
            if bs.fillet_segments > 0:
                edit_box.prop(bs, "fillet_profile")
            edit_box.prop(bs, "enable_subd_fit")

            edit_box.separator()
            edit_box.prop(bs, "auto_update")
            edit_box.prop(bs, "show_sources_wire")

            row_act = edit_box.row(align=True)
            row_act.operator("booler.rebuild", text="Rebuild", icon="FILE_REFRESH")
            row_act.operator("booler.apply", text="Apply / Bake", icon="CHECKMARK")

            if bs.last_summary:
                stat_box = layout.box()
                stat_box.label(text=bs.last_summary, icon="INFO")


CLASSES = (BOOLER_PT_main,)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
