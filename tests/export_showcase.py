"""Generate a visual showcase .blend comparing Raw Exact Boolean vs Booler (Sharp & Fillet) with SubD Lv.2.

Usage:
    blender --background --factory-startup --python tests/export_showcase.py
"""

from __future__ import annotations

import os
import sys

import bpy

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(REPO_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from Booler.core.boolean_cut import execute_boolean_cut
from Booler.core.pipeline import BoolerConfig, run_booler_pipeline
from Booler.tests.test_cases import ALL_CASES


def _add_subd(obj: bpy.types.Object, levels: int = 2) -> None:
    mod = obj.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels


def main() -> None:
    out_path = os.path.join(REPO_DIR, "tests", "showcase_preview.blend")

    # Build a clean master scene holding all 5 cases laid out on a grid
    # Row Y = 0: Raw Exact Boolean (+ SubD)
    # Row Y = 3.5: Booler Sharp (+ SubD)
    # Row Y = 7.0: Booler C1 Fillet (segments=2, + SubD)
    collected: list[tuple[str, bpy.types.Mesh, tuple[float, float, float]]] = []

    for col_idx, (case_name, builder) in enumerate(ALL_CASES):
        x_off = col_idx * 3.5
        obj_a, obj_b, op = builder()

        # 1. Raw Exact Boolean
        raw = execute_boolean_cut(obj_a, obj_b, operation=op, solver="EXACT")
        me_raw = bpy.data.meshes.new(f"{case_name}_RawBool")
        me_raw.use_fake_user = True
        raw.bm.to_mesh(me_raw)
        raw.bm.free()
        collected.append((f"{case_name}_RawBool", me_raw, (x_off, 0.0, 0.0)))

        # 2. Booler Sharp + SubD Limit Fit
        res_sharp, _ = run_booler_pipeline(
            obj_a, obj_b,
            BoolerConfig(operation=op, solver="EXACT", fillet_segments=0, enable_subd_fit=True),
            result_name=f"{case_name}_BoolerSharp",
        )
        me_sharp = res_sharp.data.copy()
        me_sharp.use_fake_user = True
        collected.append((f"{case_name}_BoolerSharp", me_sharp, (x_off, 3.5, 0.0)))

        # 3. Booler C1 Fillet (segments=2)
        res_fillet, _ = run_booler_pipeline(
            obj_a, obj_b,
            BoolerConfig(operation=op, solver="EXACT", fillet_segments=2, fillet_profile=0.5, enable_subd_fit=True),
            result_name=f"{case_name}_BoolerFillet",
        )
        me_fillet = res_fillet.data.copy()
        me_fillet.use_fake_user = True
        collected.append((f"{case_name}_BoolerFillet", me_fillet, (x_off, 7.0, 0.0)))

    # Clear temporary objects and assemble showcase scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for name, me, loc in collected:
        obj = bpy.data.objects.new(name, me)
        obj.location = loc
        bpy.context.collection.objects.link(obj)
        _add_subd(obj, levels=2)

    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"Saved showcase to: {out_path} ({len(collected)} objects)")


if __name__ == "__main__":
    main()
