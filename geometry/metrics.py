"""Topology and SubD quality metrics for evaluating Booler results."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


@dataclass
class MeshQualityReport:
    name: str
    vert_count: int
    edge_count: int
    face_count: int
    quad_count: int
    tri_count: int
    ngon_count: int
    non_manifold_edges: int
    non_manifold_verts: int
    intersection_edges: int
    intersection_irregular_poles: int
    valence_2_verts: int
    valence_3_verts: int
    valence_5_verts: int
    valence_6plus_verts: int
    min_quad_angle_deg: float
    max_quad_aspect_ratio: float
    preserved_orig_face_ratio: float
    subd_lv2_mean_error: Optional[float] = None
    subd_lv2_max_error: Optional[float] = None

    @property
    def is_all_quad(self) -> bool:
        return self.tri_count == 0 and self.ngon_count == 0 and self.face_count > 0

    @property
    def is_watertight(self) -> bool:
        return self.non_manifold_edges == 0 and self.non_manifold_verts == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_all_quad"] = self.is_all_quad
        d["is_watertight"] = self.is_watertight
        return d

    def summary_line(self) -> str:
        err_str = (
            f"subd_err={self.subd_lv2_mean_error:.5f}/{self.subd_lv2_max_error:.5f}"
            if self.subd_lv2_mean_error is not None
            else "subd_err=N/A"
        )
        return (
            f"[{self.name}] F={self.face_count} (Q={self.quad_count}, T={self.tri_count}, N={self.ngon_count}) | "
            f"NM_e={self.non_manifold_edges} NM_v={self.non_manifold_verts} | "
            f"inter_e={self.intersection_edges} inter_poles={self.intersection_irregular_poles} | "
            f"v2={self.valence_2_verts} v6+={self.valence_6plus_verts} | "
            f"minAng={self.min_quad_angle_deg:.1f}deg maxAR={self.max_quad_aspect_ratio:.2f} | "
            f"keep={self.preserved_orig_face_ratio*100:.1f}% | {err_str}"
        )


def _quad_quality(face: bmesh.types.BMFace) -> tuple[float, float]:
    """Return (min_angle_deg, aspect_ratio) for a quad face."""
    verts = [v.co for v in face.verts]
    n = len(verts)
    if n != 4:
        return 180.0, 1.0

    edge_lens = [(verts[(i + 1) % 4] - verts[i]).length for i in range(4)]
    min_len = min(edge_lens)
    max_len = max(edge_lens)
    aspect = (max_len / min_len) if min_len > 1e-9 else float("inf")

    min_angle = 180.0
    for i in range(4):
        prev_v = verts[(i - 1) % 4] - verts[i]
        next_v = verts[(i + 1) % 4] - verts[i]
        if prev_v.length > 1e-9 and next_v.length > 1e-9:
            ang = math.degrees(prev_v.angle(next_v))
            if ang < min_angle:
                min_angle = ang
        else:
            min_angle = 0.0
    return min_angle, aspect


def _compute_subd_surface_error(
    obj: bpy.types.Object,
    ref_bvh: BVHTree,
    subd_levels: int = 2,
) -> tuple[float, float]:
    """Apply temporary Catmull-Clark SubD and measure distance to reference BVH."""
    tmp_obj = obj.copy()
    tmp_obj.data = obj.data.copy()
    bpy.context.collection.objects.link(tmp_obj)
    try:
        mod = tmp_obj.modifiers.new("BoolerEvalSubD", "SUBSURF")
        mod.levels = subd_levels
        mod.render_levels = subd_levels
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = tmp_obj.evaluated_get(depsgraph)
        eval_me = eval_obj.to_mesh()
        try:
            mw = eval_obj.matrix_world
            dists = []
            for v in eval_me.vertices:
                co_w = mw @ v.co
                _, _, _, dist = ref_bvh.find_nearest(co_w)
                if dist is not None:
                    dists.append(dist)
            if not dists:
                return 0.0, 0.0
            return sum(dists) / len(dists), max(dists)
        finally:
            eval_obj.to_mesh_clear()
    finally:
        me = tmp_obj.data
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        bpy.data.meshes.remove(me, do_unlink=True)


def evaluate_mesh(
    obj: bpy.types.Object,
    name: str = "mesh",
    orig_total_kept_faces: Optional[int] = None,
    ref_bvh: Optional[BVHTree] = None,
    subd_levels: int = 2,
) -> MeshQualityReport:
    """Compute topology and SubD quality metrics on a Blender mesh object."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    try:
        quad_count = 0
        tri_count = 0
        ngon_count = 0
        min_ang = 180.0
        max_ar = 1.0

        layer_obj = bm.faces.layers.int.get("booler_orig_obj")
        layer_preserved = bm.faces.layers.int.get("booler_preserved")

        preserved_faces = 0
        for f in bm.faces:
            nv = len(f.verts)
            if nv == 4:
                quad_count += 1
                ang, ar = _quad_quality(f)
                if ang < min_ang:
                    min_ang = ang
                if ar > max_ar:
                    max_ar = ar
            elif nv == 3:
                tri_count += 1
            else:
                ngon_count += 1

            if layer_preserved is not None and f[layer_preserved] == 1:
                preserved_faces += 1

        nm_edges = sum(1 for e in bm.edges if not e.is_manifold)
        nm_verts = sum(1 for v in bm.verts if not v.is_manifold)

        inter_edges = []
        inter_verts = set()
        if layer_obj is not None:
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    if e.link_faces[0][layer_obj] != e.link_faces[1][layer_obj]:
                        inter_edges.append(e)
                        inter_verts.add(e.verts[0])
                        inter_verts.add(e.verts[1])

        inter_irregular = sum(1 for v in inter_verts if len(v.link_edges) != 4)

        v2 = v3 = v5 = v6p = 0
        for v in bm.verts:
            val = len(v.link_edges)
            if val == 2:
                v2 += 1
            elif val == 3:
                v3 += 1
            elif val == 5:
                v5 += 1
            elif val >= 6:
                v6p += 1

        denom = orig_total_kept_faces if orig_total_kept_faces else max(len(bm.faces), 1)
        preserved_ratio = preserved_faces / denom

        mean_err = max_err = None
        if ref_bvh is not None:
            mean_err, max_err = _compute_subd_surface_error(obj, ref_bvh, subd_levels=subd_levels)

        return MeshQualityReport(
            name=name,
            vert_count=len(bm.verts),
            edge_count=len(bm.edges),
            face_count=len(bm.faces),
            quad_count=quad_count,
            tri_count=tri_count,
            ngon_count=ngon_count,
            non_manifold_edges=nm_edges,
            non_manifold_verts=nm_verts,
            intersection_edges=len(inter_edges),
            intersection_irregular_poles=inter_irregular,
            valence_2_verts=v2,
            valence_3_verts=v3,
            valence_5_verts=v5,
            valence_6plus_verts=v6p,
            min_quad_angle_deg=min_ang if quad_count > 0 else 0.0,
            max_quad_aspect_ratio=max_ar if quad_count > 0 else 0.0,
            preserved_orig_face_ratio=preserved_ratio,
            subd_lv2_mean_error=mean_err,
            subd_lv2_max_error=max_err,
        )
    finally:
        bm.free()


def build_reference_bvh(obj: bpy.types.Object) -> BVHTree:
    """Build a world-space BVHTree from a mesh object (triangulated via bmesh)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh
