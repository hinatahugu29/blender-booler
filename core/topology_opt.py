"""Stage 4: Local topology optimization and tangential surface relaxation."""

from __future__ import annotations

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .boolean_cut import ATTR_ORIG_OBJ, ATTR_PRESERVED


def _classify_local_verts(
    bm: bmesh.types.BMesh,
) -> tuple[set[bmesh.types.BMVert], set[bmesh.types.BMVert], dict[bmesh.types.BMVert, int]]:
    """Classify vertices into fixed original vertices, intersection vertices, and movable local vertices."""
    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)

    fixed_orig: set[bmesh.types.BMVert] = set()
    inter_verts: set[bmesh.types.BMVert] = set()
    vert_side: dict[bmesh.types.BMVert, int] = {}

    for v in bm.verts:
        sides = set()
        touches_preserved = False
        for f in v.link_faces:
            if layer_preserved is not None and f[layer_preserved] == 1:
                touches_preserved = True
            if layer_obj is not None:
                sides.add(f[layer_obj])

        if touches_preserved:
            fixed_orig.add(v)
        if len(sides) >= 2:
            inter_verts.add(v)
        elif len(sides) == 1:
            vert_side[v] = next(iter(sides))

    return fixed_orig, inter_verts, vert_side


def _cleanup_valence2_verts(bm: bmesh.types.BMesh, fixed_verts: set[bmesh.types.BMVert]) -> int:
    """Collapse any stray valence-2 vertices in the local region (Connectivity Editing safety net)."""
    collapsed = 0
    for v in list(bm.verts):
        if not v.is_valid or v in fixed_verts:
            continue
        if len(v.link_edges) == 2 and len(v.link_faces) == 2:
            # Dissolve valence-2 vertex if safe
            try:
                bmesh.ops.dissolve_verts(bm, verts=[v])
                collapsed += 1
            except Exception:
                pass
    return collapsed


def optimize_local_region(
    bm: bmesh.types.BMesh,
    bvh_a: BVHTree,
    bvh_b: BVHTree,
    min_collar_clearance: float = 0.015,
    iterations: int = 8,
    relax_factor: float = 0.5,
) -> int:
    """Relax movable local vertices (collar rim & bridge interior) along original surfaces.

    Untouched original mesh vertices and intersection curve vertices remain strictly fixed.
    Returns number of relaxed vertices.
    """
    fixed_orig, inter_verts, vert_side = _classify_local_verts(bm)
    fixed_all = fixed_orig | inter_verts

    _cleanup_valence2_verts(bm, fixed_all)

    # Re-classify after any topology cleanup
    fixed_orig, inter_verts, vert_side = _classify_local_verts(bm)
    fixed_all = fixed_orig | inter_verts

    movable = [v for v in bm.verts if v not in fixed_all and v in vert_side]
    if not movable:
        return 0

    # Identify collar rim vertices (movable vertices directly connected to an intersection vertex)
    collar_rim: dict[bmesh.types.BMVert, list[bmesh.types.BMVert]] = {}
    for v in movable:
        adj_inter = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) in inter_verts]
        if adj_inter:
            collar_rim[v] = adj_inter

    for _ in range(iterations):
        updates: dict[bmesh.types.BMVert, Vector] = {}
        for v in movable:
            bvh = bvh_a if vert_side[v] == 1 else bvh_b
            nbrs = [e.other_vert(v) for e in v.link_edges]
            if not nbrs:
                continue

            # Weighted Laplacian: give slightly less pull from intersection verts so collar width stays healthy
            weighted_sum = Vector((0.0, 0.0, 0.0))
            w_total = 0.0
            for nv in nbrs:
                w = 0.5 if nv in inter_verts else 1.0
                weighted_sum += nv.co * w
                w_total += w
            target = weighted_sum / max(w_total, 1e-9)
            cand = v.co.lerp(target, relax_factor)

            # Project onto original surface
            loc, _, _, _ = bvh.find_nearest(cand)
            if loc is not None:
                cand = loc.copy()

            # Guard collar rim vertices from collapsing too close to intersection curve
            if v in collar_rim:
                anchor = collar_rim[v][0].co
                if (cand - anchor).length < min_collar_clearance:
                    dir_v = cand - anchor
                    if dir_v.length > 1e-8:
                        pushed = anchor + dir_v.normalized() * min_collar_clearance
                        loc2, _, _, _ = bvh.find_nearest(pushed)
                        cand = loc2.copy() if loc2 is not None else pushed
                    else:
                        cand = v.co

            updates[v] = cand

        for v, pos in updates.items():
            v.co = pos

    bm.normal_update()
    return len(movable)
