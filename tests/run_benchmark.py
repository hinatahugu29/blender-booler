"""Headless Blender benchmark runner for Booler.

Usage:
    blender --background --factory-startup --python tests/run_benchmark.py
"""

from __future__ import annotations

import os
import sys

# Ensure the parent directory of Booler is importable so Booler acts as a package
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(REPO_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from Booler.core.boolean_cut import execute_boolean_cut
from Booler.core.pipeline import BoolerConfig, run_booler_pipeline
from Booler.geometry.metrics import build_reference_bvh, evaluate_mesh
from Booler.tests.test_cases import ALL_CASES


def main() -> None:
    print("=" * 100)
    print("BOOLER BENCHMARK SUITE")
    print("=" * 100)

    for case_name, builder in ALL_CASES:
        obj_a, obj_b, op = builder()

        # Build ground-truth reference BVH from raw Exact Boolean
        raw_cut = execute_boolean_cut(obj_a, obj_b, operation=op, solver="EXACT")
        import bpy
        ref_me = bpy.data.meshes.new(f"{case_name}_Ref")
        raw_cut.bm.to_mesh(ref_me)
        raw_cut.bm.free()
        ref_obj = bpy.data.objects.new(f"{case_name}_Ref", ref_me)
        ref_obj.matrix_world = obj_a.matrix_world.copy()
        bpy.context.collection.objects.link(ref_obj)
        ref_bvh = build_reference_bvh(ref_obj)

        # Run Booler pipeline
        cfg = BoolerConfig(operation=op, solver="EXACT", enable_collar=True)
        res_obj, collar_res = run_booler_pipeline(obj_a, obj_b, config=cfg, result_name=f"{case_name}_Res")

        report = evaluate_mesh(
            res_obj,
            name=case_name,
            orig_total_kept_faces=raw_cut.total_face_count,
            ref_bvh=ref_bvh,
            subd_levels=2,
        )
        print(report.summary_line())
        if collar_res is not None:
            gap_desc = ", ".join(
                f"s{g.side_id}(in={len(g.inner_verts)},out={len(g.outer_verts)},sum={(len(g.inner_verts)+len(g.outer_verts))%2==0})"
                for g in collar_res.gap_pairs
            )
            print(
                f"   -> cutout={collar_res.cutout_face_count}, collar_quads={collar_res.collar_face_count}, "
                f"gaps[{len(collar_res.gap_pairs)}]: {gap_desc}"
            )

    print("=" * 100)


if __name__ == "__main__":
    main()
