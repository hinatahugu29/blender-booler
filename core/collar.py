"""Stage 2: Intersection resampling, local cutout, and quad collar (support strip) generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .boolean_cut import ATTR_ORIG_OBJ, ATTR_PRESERVED, BooleanCutResult, IntersectionLoop

# Max stretch factor applied to collar offset directions at sharp corners.
_CORNER_OFFSET_MAX = 1.35
# Minimum bridge gap between the collar rim and the preserved boundary, in local edge lengths.
CUTOUT_GAP_RATIO = 0.5


@dataclass
class GapPair:
    """A pair of boundary loops (inner collar rim & outer original mesh rim) to be quadrangulated."""
    side_id: int  # 1 (obj_a) or 2 (obj_b)
    inner_verts: list[bmesh.types.BMVert]  # Ordered loop along the outer edge of the collar
    outer_verts: list[bmesh.types.BMVert]  # Ordered loop along the untouched original mesh boundary
    is_closed: bool = True


@dataclass
class CapLoop:
    """An inner collar rim loop whose entire side interior was cut out (needs an all-quad cap)."""
    side_id: int
    verts: list[bmesh.types.BMVert]


@dataclass
class CollarResult:
    """Output of Stage 2 collar generation and local cutout."""
    bm: bmesh.types.BMesh
    gap_pairs: list[GapPair] = field(default_factory=list)
    cap_loops: list[CapLoop] = field(default_factory=list)
    collar_face_count: int = 0
    cutout_face_count: int = 0


def _detect_corners(coords: list[Vector], is_closed: bool, angle_deg: float = 30.0) -> list[int]:
    """Return indices of sharp corner vertices along a polyline."""
    n = len(coords)
    if n < 3:
        return list(range(n))
    thresh = math.radians(angle_deg)
    corners: list[int] = []
    for i in range(n):
        if not is_closed and (i == 0 or i == n - 1):
            corners.append(i)
            continue
        p_prev = coords[(i - 1) % n]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        if v1.length < 1e-8 or v2.length < 1e-8:
            continue
        if v1.angle(v2) >= thresh:
            corners.append(i)
    return corners


def _resample_polyline_segment(pts: list[Vector], n_segs: int) -> list[Vector]:
    """Resample an open polyline `pts` into `n_segs` equal-arc-length segments (returns n_segs + 1 points)."""
    if n_segs <= 1 or len(pts) < 2:
        return [pts[0].copy(), pts[-1].copy()]
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + (pts[i + 1] - pts[i]).length)
    total = cum[-1]
    if total < 1e-9:
        return [pts[0].copy() for _ in range(n_segs + 1)]

    out = [pts[0].copy()]
    seg_idx = 0
    for k in range(1, n_segs):
        target = total * (k / n_segs)
        while seg_idx < len(cum) - 2 and cum[seg_idx + 1] < target:
            seg_idx += 1
        seg_len = cum[seg_idx + 1] - cum[seg_idx]
        t = (target - cum[seg_idx]) / seg_len if seg_len > 1e-9 else 0.0
        out.append(pts[seg_idx].lerp(pts[seg_idx + 1], t))
    out.append(pts[-1].copy())
    return out


def resample_intersection_loop(
    coords: list[Vector],
    is_closed: bool,
    target_edge_len: float,
    force_even_count: bool = True,
    corner_angle_deg: float = 30.0,
    target_total_count: int | None = None,
) -> tuple[list[Vector], set[int]]:
    """Resample an intersection loop preserving sharp corners and enforcing even edge count if closed."""
    n = len(coords)
    if n < 2:
        return [c.copy() for c in coords], set()

    corners = _detect_corners(coords, is_closed=is_closed, angle_deg=corner_angle_deg)

    # Split into segments between corners
    segments: list[list[Vector]] = []
    if is_closed:
        if not corners:
            pts = coords + [coords[0]]
            segments = [pts]
        else:
            for idx_i in range(len(corners)):
                a = corners[idx_i]
                b = corners[(idx_i + 1) % len(corners)]
                seg = []
                curr = a
                while True:
                    seg.append(coords[curr])
                    if curr == b and len(seg) > 1:
                        break
                    curr = (curr + 1) % n
                segments.append(seg)
    else:
        for idx_i in range(len(corners) - 1):
            a = corners[idx_i]
            b = corners[idx_i + 1]
            segments.append(coords[a : b + 1])

    # Determine segment subdivision counts
    seg_lens = [
        sum((seg[i + 1] - seg[i]).length for i in range(len(seg) - 1))
        for seg in segments
    ]
    total_len = sum(seg_lens)

    if is_closed and target_total_count is not None and target_total_count >= max(4, len(segments)):
        eff_total = target_total_count
        if force_even_count and eff_total % 2 != 0:
            eff_total += 1
        # Distribute eff_total proportionally to segment lengths (at least 1 per segment)
        seg_counts = [
            max(1, int(round(eff_total * (sl / max(total_len, 1e-9)))))
            for sl in seg_lens
        ]
        # Adjust sum to match eff_total exactly
        while sum(seg_counts) < eff_total:
            best_i = max(range(len(segments)), key=lambda i: seg_lens[i] / seg_counts[i])
            seg_counts[best_i] += 1
        while sum(seg_counts) > eff_total:
            cands = [i for i in range(len(segments)) if seg_counts[i] > 1]
            if not cands:
                break
            worst_i = min(cands, key=lambda i: seg_lens[i] / seg_counts[i])
            seg_counts[worst_i] -= 1
    else:
        seg_counts = []
        for sl in seg_lens:
            c = max(1, int(round(sl / max(target_edge_len, 1e-4))))
            if is_closed and len(segments) == 1:
                c = max(4, c)
            seg_counts.append(c)

    if is_closed and force_even_count and (sum(seg_counts) % 2 != 0):
        best_idx = max(range(len(segments)), key=lambda i: seg_lens[i] / seg_counts[i])
        seg_counts[best_idx] += 1

    # Rebuild resampled coordinates and track new corner indices
    resampled: list[Vector] = []
    new_corners: set[int] = set()
    for seg_i, (seg, cnt) in enumerate(zip(segments, seg_counts)):
        pts = _resample_polyline_segment(seg, cnt)
        if corners:
            new_corners.add(len(resampled))
        if is_closed or seg_i < len(segments) - 1:
            resampled.extend(pts[:-1])
        else:
            resampled.extend(pts)

    return resampled, new_corners


def _estimate_local_edge_length(
    bm: bmesh.types.BMesh,
    inter_verts: set[bmesh.types.BMVert],
    side_id: int | None = None,
) -> float:
    """Estimate median edge length of original faces adjacent to the intersection.

    With `side_id`, only faces belonging to that source object are sampled.
    """
    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ) if side_id is not None else None
    lens = []
    seen_faces = set()
    for v in inter_verts:
        for f in v.link_faces:
            for e in f.edges:
                for nf in e.link_faces:
                    if layer_obj is not None and nf[layer_obj] != side_id:
                        continue
                    if nf not in seen_faces:
                        seen_faces.add(nf)
                        for ne in nf.edges:
                            if ne.verts[0] not in inter_verts and ne.verts[1] not in inter_verts:
                                lens.append(ne.calc_length())
    if not lens:
        lens = [e.calc_length() for e in bm.edges if e.calc_length() > 1e-6]
    if not lens:
        return 0.2
    lens.sort()
    return lens[len(lens) // 2]  # Median is robust against sliver edges


def _compute_side_offset_points(
    resampled: list[Vector],
    is_closed: bool,
    corner_indices: set[int],
    side_id: int,
    raw_bm: bmesh.types.BMesh,
    side_bvh: BVHTree,
    width: float,
    rings: int = 1,
) -> list[list[Vector]]:
    """Generate `rings` offset curves on `side_id`'s surface away from the intersection loop."""
    n = len(resampled)
    layer_obj = raw_bm.faces.layers.int.get(ATTR_ORIG_OBJ)

    # Build a temporary BVH of ONLY the kept faces belonging to `side_id` in the raw boolean result
    side_bm = bmesh.new()
    vert_map: dict[int, bmesh.types.BMVert] = {}
    side_face_centers: list[Vector] = []
    side_face_normals: list[Vector] = []
    for f in raw_bm.faces:
        if layer_obj is not None and f[layer_obj] == side_id:
            fverts = []
            for v in f.verts:
                nv = vert_map.get(v.index)
                if nv is None:
                    nv = side_bm.verts.new(v.co)
                    vert_map[v.index] = nv
                fverts.append(nv)
            try:
                nf = side_bm.faces.new(fverts)
                side_face_centers.append(f.calc_center_median())
                side_face_normals.append(f.normal.copy())
            except ValueError:
                pass
    bmesh.ops.triangulate(side_bm, faces=side_bm.faces)
    kept_bvh = BVHTree.FromBMesh(side_bm)
    side_bm.free()

    # Compute outward-along-surface offset direction at each resampled point
    dirs: list[Vector] = []
    for i in range(n):
        p = resampled[i]
        p_prev = resampled[(i - 1) % n] if is_closed else resampled[max(0, i - 1)]
        p_next = resampled[(i + 1) % n] if is_closed else resampled[min(n - 1, i + 1)]
        v_in = p - p_prev
        v_out = p_next - p
        t_in = v_in.normalized() if v_in.length > 1e-8 else Vector((1, 0, 0))
        t_out = v_out.normalized() if v_out.length > 1e-8 else t_in
        tangent = t_in + t_out
        if tangent.length < 1e-8:
            tangent = t_in
        tangent.normalize()

        # Find surface normal by sampling slightly along the segment midpoint if at a sharp corner
        sample_pt = p
        if i in corner_indices:
            sample_pt = p.lerp(p_next, 0.15)
        loc, normal, _, _ = kept_bvh.find_nearest(sample_pt)
        if normal is None or normal.length < 1e-8:
            loc, normal, _, _ = side_bvh.find_nearest(sample_pt)
        if normal is None or normal.length < 1e-8:
            normal = Vector((0, 0, 1))
        normal = normal.normalized()

        # Tangent-plane direction perpendicular to loop tangent
        cand = normal.cross(tangent)
        if cand.length < 1e-8:
            cand = Vector((0, 0, 1)).cross(tangent)
        cand.normalize()

        # Robust sign disambiguation: test multiple probe distances along +cand vs -cand
        score_pos = 0.0
        score_neg = 0.0
        for scale in (0.15, 0.4, 0.8):
            step = max(width * scale, 1e-3)
            _, _, _, dp = kept_bvh.find_nearest(p + cand * step)
            _, _, _, dn = kept_bvh.find_nearest(p - cand * step)
            score_pos += dp if dp is not None else 1e6
            score_neg += dn if dn is not None else 1e6
        if score_neg < score_pos:
            cand = -cand

        # For sharp corners, average the two adjacent segment offset directions to avoid collapse
        if i in corner_indices and is_closed:
            _, n_in, _, _ = kept_bvh.find_nearest(p.lerp(p_prev, 0.2))
            _, n_out, _, _ = kept_bvh.find_nearest(p.lerp(p_next, 0.2))
            c_list = []
            for seg_t, seg_n in ((t_in, n_in), (t_out, n_out)):
                if seg_n is not None and seg_n.length > 1e-8:
                    c = seg_n.normalized().cross(seg_t)
                    if c.length > 1e-8:
                        c.normalize()
                        if c.dot(cand) < 0:
                            c = -c
                        c_list.append(c)
            if len(c_list) == 2:
                avg_c = c_list[0] + c_list[1]
                if avg_c.length > 1e-6:
                    avg_c.normalize()
                    cos_half = max(0.65, c_list[0].dot(avg_c))
                    cand = avg_c * min(_CORNER_OFFSET_MAX, 1.0 / cos_half)

        dirs.append(cand)

    # Generate offset rings projected onto the kept surface
    all_rings: list[list[Vector]] = []
    for r in range(1, rings + 1):
        step_w = width * (r / rings)
        ring_pts: list[Vector] = []
        for i in range(n):
            raw_pt = resampled[i] + dirs[i] * step_w
            loc, _, _, _ = kept_bvh.find_nearest(raw_pt)
            if loc is None:
                loc, _, _, _ = side_bvh.find_nearest(raw_pt)
            pt = loc.copy() if loc is not None else raw_pt
            # Guard against projection collapsing back onto the intersection curve
            if (pt - resampled[i]).length < step_w * 0.4:
                # Re-try using full side_bvh from further out
                loc2, _, _, _ = side_bvh.find_nearest(raw_pt)
                if loc2 is not None and (loc2 - resampled[i]).length > (pt - resampled[i]).length:
                    pt = loc2.copy()
                else:
                    pt = raw_pt
            ring_pts.append(pt)

        # Light tangential smoothing + re-projection (preserve corners)
        for _ in range(2):
            smoothed = []
            for i in range(n):
                if i in corner_indices or (not is_closed and (i == 0 or i == n - 1)):
                    smoothed.append(ring_pts[i])
                    continue
                avg = (ring_pts[(i - 1) % n] + ring_pts[i] * 2.0 + ring_pts[(i + 1) % n]) * 0.25
                loc, _, _, _ = kept_bvh.find_nearest(avg)
                pt = loc.copy() if loc is not None else avg
                if (pt - resampled[i]).length < step_w * 0.35:
                    pt = ring_pts[i]
                smoothed.append(pt)
            ring_pts = smoothed

        all_rings.append(ring_pts)

    return all_rings


def _extract_ordered_boundary_loops(
    boundary_edges: list[bmesh.types.BMEdge],
) -> list[list[bmesh.types.BMVert]]:
    """Extract ordered vertex loops from a set of boundary edges."""
    if not boundary_edges:
        return []
    adj: dict[bmesh.types.BMVert, list[bmesh.types.BMEdge]] = {}
    for e in boundary_edges:
        adj.setdefault(e.verts[0], []).append(e)
        adj.setdefault(e.verts[1], []).append(e)

    visited: set[bmesh.types.BMEdge] = set()
    loops: list[list[bmesh.types.BMVert]] = []

    for start_e in boundary_edges:
        if start_e in visited:
            continue
        start_v = start_e.verts[0]
        curr_v = start_e.other_vert(start_v)
        visited.add(start_e)
        verts = [start_v, curr_v]

        while curr_v != start_v:
            nxt = [e for e in adj.get(curr_v, []) if e not in visited]
            if not nxt:
                break
            e = nxt[0]
            visited.add(e)
            curr_v = e.other_vert(curr_v)
            if curr_v != start_v:
                verts.append(curr_v)
        loops.append(verts)

    return loops


def _build_curve_kdtree(
    loops: list[tuple[bool, list[Vector]]],
    spacing: float,
) -> KDTree | None:
    """Densely sample intersection polylines into a KDTree for point-to-curve distance queries.

    Nearest-sample distance over-estimates the true point-to-polyline distance by at most
    `spacing / 2`.
    """
    spacing = max(spacing, 1e-6)
    pts: list[Vector] = []
    for is_closed, coords in loops:
        if not coords:
            continue
        nxt = coords[1:] + ([coords[0]] if is_closed else [])
        for a, b in zip(coords, nxt):
            k = max(1, int(math.ceil((b - a).length / spacing)))
            for t in range(k):
                pts.append(a.lerp(b, t / k))
        if not is_closed:
            pts.append(coords[-1].copy())
    if not pts:
        return None
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    return kd


def _expand_and_clean_cutout(
    bm: bmesh.types.BMesh,
    inter_verts: set[bmesh.types.BMVert],
    curve_kd: KDTree | None,
    clearance_by_side: dict[int, float],
) -> set[bmesh.types.BMFace]:
    """Select faces to remove around the intersection so the remaining boundary has clean clearance.

    Every vertex of a kept face ends up at least its side's clearance away from the intersection
    curve, which guarantees the gap between the collar rim and the preserved boundary has real
    width (otherwise the bridge quads degenerate into slivers or bowties).
    """
    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    default_clearance = min(clearance_by_side.values()) if clearance_by_side else 0.0
    to_remove: set[bmesh.types.BMFace] = set()

    # 1. Always remove faces touching intersection vertices (they are cut/N-gon/tri faces)
    for v in inter_verts:
        for f in v.link_faces:
            to_remove.add(f)

    # 2. Expand to any face having a vertex closer than `clearance` to the intersection curve
    if curve_kd is not None:
        curve_dist: dict[bmesh.types.BMVert, float] = {}
        for f in bm.faces:
            if f in to_remove:
                continue
            sid = f[layer_obj] if layer_obj is not None else None
            clearance = clearance_by_side.get(sid, default_clearance)
            for v in f.verts:
                d = curve_dist.get(v)
                if d is None:
                    d = curve_kd.find(v.co)[2]
                    curve_dist[v] = d
                if d < clearance:
                    to_remove.add(f)
                    break

    # 3. Clean up protruding "ear" faces (kept faces with >= 3 edges adjacent to removed region)
    #    and pinch vertices (kept vertices with > 2 boundary edges)
    changed = True
    while changed:
        changed = False
        for f in list(bm.faces):
            if f in to_remove:
                continue
            adj_removed = 0
            for e in f.edges:
                if len(e.link_faces) == 2:
                    other = e.link_faces[0] if e.link_faces[1] == f else e.link_faces[1]
                    if other in to_remove:
                        adj_removed += 1
            if adj_removed >= 3:
                to_remove.add(f)
                changed = True

        # Check for pinch vertices (a kept vertex whose incident boundary edges > 2)
        if not changed:
            bnd_count: dict[bmesh.types.BMVert, int] = {}
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    r0 = e.link_faces[0] in to_remove
                    r1 = e.link_faces[1] in to_remove
                    if r0 != r1:
                        bnd_count[e.verts[0]] = bnd_count.get(e.verts[0], 0) + 1
                        bnd_count[e.verts[1]] = bnd_count.get(e.verts[1], 0) + 1
            for v, cnt in bnd_count.items():
                if cnt > 2:
                    # Absorb the kept faces around this pinch vertex
                    for f in v.link_faces:
                        if f not in to_remove:
                            to_remove.add(f)
                            changed = True

    return to_remove


def build_collar_and_cutout(
    cut_res: BooleanCutResult,
    bvh_a: BVHTree,
    bvh_b: BVHTree,
    collar_width: float = 0.05,
    collar_rings: int = 1,
    target_edge_len: float | None = None,
) -> CollarResult:
    """Execute Stage 2: resample intersection loops, cut out damaged faces, and build quad collars."""
    bm = cut_res.bm
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)

    inter_verts = {v for e in cut_res.inter_edges for v in e.verts}
    if target_edge_len is None or target_edge_len <= 0.0:
        target_edge_len = _estimate_local_edge_length(bm, inter_verts)

    # Clamp collar width relative to local edge length so it doesn't swallow the whole mesh
    eff_width = min(collar_width, target_edge_len * 0.45)

    raw_loop_coords = [(loop.is_closed, [v.co.copy() for v in loop.verts]) for loop in cut_res.inter_loops]

    # 1. Identify cutout faces (without deleting yet, so Pass 2 can still query kept faces in bm).
    # Kept boundary must clear the widest possible collar rim (corner offsets stretch up to
    # _CORNER_OFFSET_MAX x width) plus a bridge gap proportional to the local edge length.
    # The gap uses each side's own edge length so a fine-meshed thin object is not swallowed
    # by a clearance derived from a coarse partner.
    clearance_by_side = {
        sid: eff_width * _CORNER_OFFSET_MAX
        + min(target_edge_len, _estimate_local_edge_length(bm, inter_verts, side_id=sid))
        * CUTOUT_GAP_RATIO
        for sid in (1, 2)
    }
    curve_kd = _build_curve_kdtree(raw_loop_coords, spacing=min(clearance_by_side.values()) * 0.1)
    to_remove = _expand_and_clean_cutout(
        bm=bm,
        inter_verts=inter_verts,
        curve_kd=curve_kd,
        clearance_by_side=clearance_by_side,
    )

    # Preview outer boundary loop lengths per side BEFORE deleting faces
    preview_outer_edges: dict[int, list[bmesh.types.BMEdge]] = {1: [], 2: []}
    for e in bm.edges:
        if len(e.link_faces) == 2 and layer_obj is not None:
            r0 = e.link_faces[0] in to_remove
            r1 = e.link_faces[1] in to_remove
            if r0 != r1:
                kept_f = e.link_faces[1] if r0 else e.link_faces[0]
                sid = kept_f[layer_obj]
                if sid in preview_outer_edges:
                    preview_outer_edges[sid].append(e)
    preview_outer_loops = {
        1: _extract_ordered_boundary_loops(preview_outer_edges[1]),
        2: _extract_ordered_boundary_loops(preview_outer_edges[2]),
    }

    # Pass 2: Match each intersection loop's subdivision count to its paired outer boundary counts
    loop_data = []
    for is_closed, coords in raw_loop_coords:
        target_cnt: int | None = None
        if is_closed and coords:
            lc = sum(coords, Vector()) / len(coords)
            matched_lens = []
            for sid in (1, 2):
                if preview_outer_loops[sid]:
                    nearest = min(
                        preview_outer_loops[sid],
                        key=lambda ol: (sum((v.co for v in ol), Vector()) / max(len(ol), 1) - lc).length,
                    )
                    matched_lens.append(len(nearest))
            if matched_lens:
                # Choose an even count close to the average/min of the two outer loops to minimize reductions
                avg_cnt = int(round(sum(matched_lens) / len(matched_lens)))
                if avg_cnt % 2 != 0:
                    avg_cnt = min(matched_lens) if min(matched_lens) % 2 == 0 else avg_cnt + 1
                target_cnt = max(4, avg_cnt)

        resampled, corners = resample_intersection_loop(
            coords=coords,
            is_closed=is_closed,
            target_edge_len=target_edge_len,
            force_even_count=True,
            target_total_count=target_cnt,
        )
        rings_a = _compute_side_offset_points(
            resampled=resampled,
            is_closed=is_closed,
            corner_indices=corners,
            side_id=1,
            raw_bm=bm,
            side_bvh=bvh_a,
            width=eff_width,
            rings=collar_rings,
        )
        rings_b = _compute_side_offset_points(
            resampled=resampled,
            is_closed=is_closed,
            corner_indices=corners,
            side_id=2,
            raw_bm=bm,
            side_bvh=bvh_b,
            width=eff_width,
            rings=collar_rings,
        )
        loop_data.append((is_closed, resampled, rings_a, rings_b))

    # Delete cutout faces now
    cutout_count = len(to_remove)
    if to_remove:
        bmesh.ops.delete(bm, geom=list(to_remove), context="FACES")

    # Clean up any loose edges / verts left behind
    loose_edges = [e for e in bm.edges if not e.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES_FACES")
    loose_verts = [v for v in bm.verts if not v.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    # 3. Extract outer boundary loops per side (from the untouched original mesh)
    outer_edges_by_side: dict[int, list[bmesh.types.BMEdge]] = {1: [], 2: []}
    for e in bm.edges:
        if e.is_boundary and len(e.link_faces) == 1 and layer_obj is not None:
            sid = e.link_faces[0][layer_obj]
            if sid in outer_edges_by_side:
                outer_edges_by_side[sid].append(e)

    outer_loops_by_side: dict[int, list[list[bmesh.types.BMVert]]] = {
        1: _extract_ordered_boundary_loops(outer_edges_by_side[1]),
        2: _extract_ordered_boundary_loops(outer_edges_by_side[2]),
    }

    # 4. Build clean quad collar strips in bm
    collar_faces_created = 0
    inner_rims_by_side: dict[int, list[tuple[list[bmesh.types.BMVert], bool]]] = {1: [], 2: []}

    def _add_strip(
        ring_a: list[bmesh.types.BMVert],
        ring_b: list[bmesh.types.BMVert],
        is_closed: bool,
        side_id: int,
    ) -> int:
        cnt = 0
        m = len(ring_a)
        seg_n = m if is_closed else m - 1
        for i in range(seg_n):
            j = (i + 1) % m
            try:
                f = bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
                if layer_obj is not None:
                    f[layer_obj] = side_id
                if layer_preserved is not None:
                    f[layer_preserved] = 0
                cnt += 1
            except ValueError:
                pass
        return cnt

    for is_closed, resampled, rings_a, rings_b in loop_data:
        center_verts = [bm.verts.new(p) for p in resampled]
        for side_id, side_rings in ((1, rings_a), (2, rings_b)):
            prev_ring = center_verts
            for r_pts in side_rings:
                curr_ring = [bm.verts.new(p) for p in r_pts]
                collar_faces_created += _add_strip(prev_ring, curr_ring, is_closed, side_id)
                prev_ring = curr_ring
            inner_rims_by_side[side_id].append((prev_ring, is_closed))

    # 5. Globally match inner collar rims to outer boundary loops per side by loop-to-loop proximity
    gap_pairs: list[GapPair] = []
    cap_loops: list[CapLoop] = []

    def _loop_pair_distance(la: list[bmesh.types.BMVert], lb: list[bmesh.types.BMVert]) -> float:
        """Mean nearest-vertex distance from `la` to `lb` (robust even when centroids differ)."""
        if not la or not lb:
            return float("inf")
        total = 0.0
        for va in la:
            p = va.co
            total += min((p - vb.co).length for vb in lb)
        return total / len(la)

    for side_id in (1, 2):
        inners = inner_rims_by_side[side_id]
        outers = outer_loops_by_side[side_id]
        scored: list[tuple[float, int, int]] = []
        for ii, (iring, is_closed) in enumerate(inners):
            for oi, oloop in enumerate(outers):
                d = _loop_pair_distance(iring, oloop)
                scored.append((d, ii, oi))
        scored.sort(key=lambda x: x[0])

        matched_in: set[int] = set()
        matched_out: set[int] = set()
        for d, ii, oi in scored:
            if ii in matched_in or oi in matched_out:
                continue
            matched_in.add(ii)
            matched_out.add(oi)
            iring, is_closed = inners[ii]
            gap_pairs.append(
                GapPair(
                    side_id=side_id,
                    inner_verts=iring,
                    outer_verts=outers[oi],
                    is_closed=is_closed,
                )
            )

        # Any remaining closed inner rim has no remaining outer mesh on that side -> cap with all-quads
        for ii, (iring, is_closed) in enumerate(inners):
            if ii not in matched_in and is_closed:
                cap_loops.append(CapLoop(side_id=side_id, verts=iring))

    bm.normal_update()
    return CollarResult(
        bm=bm,
        gap_pairs=gap_pairs,
        cap_loops=cap_loops,
        collar_face_count=collar_faces_created,
        cutout_face_count=cutout_count,
    )
