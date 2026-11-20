"""Stage 5: C1 Hermite filleting and Catmull-Clark Limit Surface Inverse Fitting (Halstead 1993)."""

from __future__ import annotations

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .boolean_cut import ATTR_ORIG_OBJ, ATTR_PRESERVED
from .topology_opt import accept_quality_move


def _eval_quadratic_bezier(p0: Vector, p1: Vector, p2: Vector, t: float) -> Vector:
    """Evaluate quadratic Bezier curve (C1 tangent continuous with p0->p1 and p1->p2)."""
    u = 1.0 - t
    return (u * u) * p0 + (2.0 * u * t) * p1 + (t * t) * p2


def apply_c1_fillet(
    bm: bmesh.types.BMesh,
    fillet_segments: int = 2,
    fillet_profile: float = 0.5,
) -> int:
    """Convert the sharp intersection crease + collar into a smooth C1 quad fillet band.

    Uses the intersection vertices as quadratic Bezier control points between Side A and Side B
    collar rim vertices, then subdivides the intersection strip into `fillet_segments` smooth rings.
    Returns number of fillet faces created/adjusted.
    """
    if fillet_segments <= 0:
        return 0

    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)
    if layer_obj is None:
        return 0

    # Find intersection edges and vertices
    inter_edges = [
        e for e in bm.edges
        if len(e.link_faces) == 2 and e.link_faces[0][layer_obj] != e.link_faces[1][layer_obj]
    ]
    if not inter_edges:
        return 0

    inter_verts = {v for e in inter_edges for v in e.verts}
    inter_edge_set = set(inter_edges)

    # For each intersection vertex, find its Side-1 and Side-2 collar rim neighbors
    triplets: dict[bmesh.types.BMVert, tuple[bmesh.types.BMVert, bmesh.types.BMVert]] = {}
    for v in inter_verts:
        v_s1 = None
        v_s2 = None
        for e in v.link_edges:
            if e in inter_edge_set:
                continue
            ov = e.other_vert(v)
            sides = {f[layer_obj] for f in e.link_faces}
            if 1 in sides and 2 not in sides:
                v_s1 = ov
            elif 2 in sides and 1 not in sides:
                v_s2 = ov
        if v_s1 is not None and v_s2 is not None:
            triplets[v] = (v_s1, v_s2)

    if not triplets:
        return 0

    # For fillet_segments == 1: simply round the intersection vertices onto the C1 Bezier midpoint
    if fillet_segments == 1:
        for v, (va, vb) in triplets.items():
            # Blend between sharp corner (v.co) and quadratic Bezier midpoint at t=0.5
            bez_mid = _eval_quadratic_bezier(va.co, v.co, vb.co, 0.5)
            v.co = v.co.lerp(bez_mid, max(0.0, min(1.0, fillet_profile * 2.0)))
        bm.normal_update()
        return len(triplets)

    # For fillet_segments >= 2: use bmesh.ops.bevel on the intersection edges with C1 profile,
    # keeping exact quad topology (since all intersection verts have valence 4).
    # First record sharp control points before beveling
    sharp_pos = {v: v.co.copy() for v in triplets}
    rim_pos = {v: (va.co.copy(), vb.co.copy()) for v, (va, vb) in triplets.items()}

    # Compute average collar span for bevel offset
    spans = [(va.co - v.co).length + (vb.co - v.co).length for v, (va, vb) in triplets.items()]
    avg_half_span = (sum(spans) / max(len(spans), 1)) * 0.25

    res = bmesh.ops.bevel(
        bm,
        geom=list(inter_edges),
        offset=avg_half_span * max(0.2, min(0.95, fillet_profile)),
        offset_type="OFFSET",
        segments=fillet_segments,
        profile=0.5,
        affect="EDGES",
        clamp_overlap=True,
    )
    new_faces = res.get("faces", [])
    for f in new_faces:
        if layer_preserved is not None:
            f[layer_preserved] = 0

    bm.normal_update()
    return len(new_faces)


def _catmull_clark_limit_point(v: bmesh.types.BMVert) -> Vector | None:
    """Compute the exact Catmull-Clark limit surface position for vertex `v` (Halstead et al. 1993).

    Formula for valence k:
        v_inf = (k^2 * v + 4 * sum(e_i) + sum(f_i)) / (k * (k + 5))
    where e_i are adjacent edge vertices and f_i are opposite diagonal vertices of incident quads.
    """
    k = len(v.link_edges)
    if k < 3 or len(v.link_faces) != k:
        return None

    edge_sum = Vector((0.0, 0.0, 0.0))
    for e in v.link_edges:
        edge_sum += e.other_vert(v).co

    diag_sum = Vector((0.0, 0.0, 0.0))
    for f in v.link_faces:
        if len(f.verts) != 4:
            return None
        # Find the vertex diagonally opposite to `v` in this quad
        fverts = list(f.verts)
        idx = fverts.index(v)
        diag_v = fverts[(idx + 2) % 4]
        diag_sum += diag_v.co

    denom = float(k * (k + 5))
    return ((k * k) * v.co + 4.0 * edge_sum + diag_sum) / denom


def apply_catmull_clark_limit_fit(
    bm: bmesh.types.BMesh,
    bvh_a: BVHTree,
    bvh_b: BVHTree,
    iterations: int = 6,
    omega: float = 0.55,
    max_disp_ratio: float = 0.15,
) -> int:
    """Adjust local generated vertices so their Catmull-Clark limit surface matches the original surfaces.

    Untouched original mesh vertices remain strictly fixed.
    Returns number of fitted vertices.
    """
    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)

    movable: list[tuple[bmesh.types.BMVert, BVHTree, float]] = []
    for v in bm.verts:
        touches_preserved = False
        sides = set()
        for f in v.link_faces:
            if layer_preserved is not None and f[layer_preserved] == 1:
                touches_preserved = True
            if layer_obj is not None:
                sides.add(f[layer_obj])
        if touches_preserved or len(sides) != 1:
            continue
        side_id = next(iter(sides))
        bvh = bvh_a if side_id == 1 else bvh_b
        avg_len = sum(e.calc_length() for e in v.link_edges) / max(len(v.link_edges), 1)
        movable.append((v, bvh, avg_len * max_disp_ratio))

    if not movable:
        return 0

    for _ in range(iterations):
        updates: list[tuple[bmesh.types.BMVert, Vector]] = []
        for v, bvh, max_step in movable:
            v_inf = _catmull_clark_limit_point(v)
            if v_inf is None:
                continue
            loc, _, _, _ = bvh.find_nearest(v_inf)
            if loc is None:
                continue
            delta = (loc - v_inf) * omega
            if delta.length > max_step and delta.length > 1e-9:
                delta = delta.normalized() * max_step
            updates.append((v, v.co + delta))

        # Shape accuracy is this stage's purpose, so only block moves that would crush a quad.
        for v, new_co in updates:
            if accept_quality_move(v, new_co, good_angle=10.0, max_aspect=20.0):
                v.co = new_co

    bm.normal_update()
    return len(movable)
