"""Main Booler pipeline orchestration (supports both new creation and non-destructive in-place update)."""

from __future__ import annotations

from dataclasses import dataclass

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

from .boolean_cut import BooleanOp, BooleanSolver, execute_boolean_cut
from .collar import CollarResult, build_collar_and_cutout
from .remesh_local import LocalRemeshResult, quadrangulate_all_gaps
from .subd_fit import apply_c1_fillet, apply_catmull_clark_limit_fit
from .topology_opt import optimize_local_region


@dataclass
class BoolerConfig:
    operation: BooleanOp = "UNION"
    solver: BooleanSolver = "EXACT"
    collar_width: float = 0.05
    collar_rings: int = 1
    fillet_segments: int = 0
    fillet_profile: float = 0.5
    enable_collar: bool = True
    enable_local_remesh: bool = True
    enable_topology_opt: bool = True
    enable_subd_fit: bool = True


def _build_local_bvh(obj: bpy.types.Object, target_space_obj: bpy.types.Object) -> BVHTree:
    """Build a triangulated BVHTree of `obj` in `target_space_obj`'s local coordinate space."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if obj != target_space_obj:
        rel_mat = target_space_obj.matrix_world.inverted() @ obj.matrix_world
        bmesh.ops.transform(bm, matrix=rel_mat, verts=bm.verts)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def _execute_to_bmesh(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    config: BoolerConfig,
) -> tuple[bmesh.types.BMesh, CollarResult | None]:
    """Run all enabled Booler stages and return the resulting BMesh."""
    # Stage 1: Exact/Manifold Boolean & intersection loop extraction
    cut_res = execute_boolean_cut(
        obj_a=obj_a,
        obj_b=obj_b,
        operation=config.operation,
        solver=config.solver,
    )

    collar_res: CollarResult | None = None
    if config.enable_collar and cut_res.inter_loops:
        bvh_a = _build_local_bvh(obj_a, obj_a)
        bvh_b = _build_local_bvh(obj_b, obj_a)
        # Stage 2: Collar / support strip generation & local ring cutout
        collar_res = build_collar_and_cutout(
            cut_res=cut_res,
            bvh_a=bvh_a,
            bvh_b=bvh_b,
            collar_width=config.collar_width,
            collar_rings=config.collar_rings,
        )
        # Stage 3: Boundary-constrained local quadrangulation
        if config.enable_local_remesh and (collar_res.gap_pairs or collar_res.cap_loops):
            quadrangulate_all_gaps(
                bm=collar_res.bm,
                gap_pairs=collar_res.gap_pairs,
                bvh_a=bvh_a,
                bvh_b=bvh_b,
                cap_loops=collar_res.cap_loops,
            )

        # Stage 4: Local topology optimization & surface relaxation
        if config.enable_topology_opt:
            optimize_local_region(
                bm=collar_res.bm,
                bvh_a=bvh_a,
                bvh_b=bvh_b,
                min_collar_clearance=config.collar_width * 0.35,
                iterations=8,
            )

        # Stage 5: Optional C1 fillet & Catmull-Clark Limit Surface Inverse Fitting
        if config.fillet_segments > 0:
            apply_c1_fillet(
                bm=collar_res.bm,
                fillet_segments=config.fillet_segments,
                fillet_profile=config.fillet_profile,
            )
        if config.enable_subd_fit:
            apply_catmull_clark_limit_fit(
                bm=collar_res.bm,
                bvh_a=bvh_a,
                bvh_b=bvh_b,
                iterations=6,
            )

    return cut_res.bm, collar_res


def run_booler_pipeline(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    config: BoolerConfig,
    result_name: str = "Booler_Result",
) -> tuple[bpy.types.Object, CollarResult | None]:
    """Execute the Booler pipeline and create a new result object in the active collection."""
    bm, collar_res = _execute_to_bmesh(obj_a, obj_b, config)
    out_mesh = bpy.data.meshes.new(result_name)
    try:
        bm.to_mesh(out_mesh)
        out_mesh.update()
    finally:
        bm.free()

    out_obj = bpy.data.objects.new(result_name, out_mesh)
    out_obj.matrix_world = obj_a.matrix_world.copy()
    bpy.context.collection.objects.link(out_obj)
    return out_obj, collar_res


def update_booler_object(
    target_obj: bpy.types.Object,
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    config: BoolerConfig,
) -> CollarResult | None:
    """Non-destructively update `target_obj.data` in-place, preserving modifiers and materials."""
    bm, collar_res = _execute_to_bmesh(obj_a, obj_b, config)
    try:
        bm.to_mesh(target_obj.data)
        target_obj.data.update()
    finally:
        bm.free()
    target_obj.matrix_world = obj_a.matrix_world.copy()
    return collar_res
