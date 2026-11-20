"""Stage 4: Local topology optimization and tangential surface relaxation."""

from __future__ import annotations

import math

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


# Moves that keep every incident face at or above this minimum angle are always accepted.
GOOD_ANGLE_DEG = 35.0
# A move may not push any incident face beyond this edge-length aspect ratio unless that face
# was already worse (thin support-loop strips legitimately sit around 5-15).
MAX_ASPECT = 10.0


def _face_quality(
    face: bmesh.types.BMFace, moved: bmesh.types.BMVert, moved_co: Vector
) -> tuple[float, float]:
    """(min corner angle in degrees, edge aspect ratio) of `face` with `moved` at `moved_co`."""
    cos = [moved_co if fv == moved else fv.co for fv in face.verts]
    n = len(cos)
    min_ang = 180.0
    lens = []
    for k in range(n):
        va = cos[k - 1] - cos[k]
        vb = cos[(k + 1) % n] - cos[k]
        if va.length < 1e-9 or vb.length < 1e-9:
            return 0.0, float("inf")
        lens.append(vb.length)
        min_ang = min(min_ang, math.degrees(va.angle(vb)))
    return min_ang, max(lens) / min(lens)


def vertex_star_quality(v: bmesh.types.BMVert, co: Vector | None = None) -> tuple[float, float]:
    """(min angle, max aspect) over all faces around `v`, optionally with `v` placed at `co`."""
    pos = v.co if co is None else co
    min_ang = 180.0
    max_ar = 1.0
    for f in v.link_faces:
        ang, ar = _face_quality(f, v, pos)
        min_ang = min(min_ang, ang)
        max_ar = max(max_ar, ar)
    return min_ang, max_ar


def accept_quality_move(
    v: bmesh.types.BMVert,
    new_co: Vector,
    good_angle: float = GOOD_ANGLE_DEG,
    max_aspect: float = MAX_ASPECT,
) -> bool:
    """Return True if moving `v` to `new_co` does not degrade its surrounding faces.

    The minimum angle must stay healthy (>= good_angle) or not get worse, and the maximum
    aspect ratio must stay within max_aspect or not get worse, so relaxation and fitting passes
    can never fold, crush or thin out quads. Looser thresholds only block severe degradation.
    """
    new_ang, new_ar = vertex_star_quality(v, new_co)
    if new_ang >= good_angle and new_ar <= max_aspect:
        return True
    old_ang, old_ar = vertex_star_quality(v)
    angle_ok = new_ang >= good_angle or new_ang >= old_ang
    aspect_ok = new_ar <= max_aspect or new_ar <= old_ar
    return angle_ok and aspect_ok


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

    # Gauss-Seidel sweeps: each vertex is moved in place only if the quality guard accepts it,
    # so every accepted move is checked against its neighbors' current positions.
    for _ in range(iterations):
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

            if accept_quality_move(v, cand):
                v.co = cand

    bm.normal_update()
    return len(movable)
