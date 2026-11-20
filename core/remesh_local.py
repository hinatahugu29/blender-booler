"""Stage 3: Boundary-constrained local quadrangulation for gap rings (3-to-1 SubD reduction bridge)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .boolean_cut import ATTR_ORIG_OBJ, ATTR_PRESERVED
from .collar import CapLoop, GapPair


@dataclass
class LocalRemeshResult:
    """Summary of Stage 3 local quadrangulation."""
    faces_created: int = 0
    internal_verts_created: int = 0
    gaps_filled: int = 0
    gaps_skipped: int = 0


def _loop_arc_params(verts: list[bmesh.types.BMVert]) -> list[float]:
    """Return normalized cumulative arc-length parameters in [0, 1) for a closed vertex loop."""
    n = len(verts)
    if n == 0:
        return []
    cum = [0.0]
    for i in range(n):
        cum.append(cum[-1] + (verts[(i + 1) % n].co - verts[i].co).length)
    total = cum[-1]
    if total < 1e-9:
        return [i / n for i in range(n)]
    return [cum[i] / total for i in range(n)]


def _align_loop_orientation_and_shift(
    loop_a: list[bmesh.types.BMVert],
    loop_b: list[bmesh.types.BMVert],
) -> list[bmesh.types.BMVert]:
    """Orient and cyclically shift `loop_b` to minimize 3D twist/distance against `loop_a`."""
    m = len(loop_a)
    n = len(loop_b)
    if m == 0 or n == 0:
        return list(loop_b)

    best_cost = float("inf")
    best_loop = list(loop_b)

    for candidate in (list(loop_b), list(reversed(loop_b))):
        for shift in range(n):
            cost = 0.0
            for i in range(m):
                j = (shift + int(round(i * n / m))) % n
                cost += (loop_a[i].co - candidate[j].co).length_squared
            if cost < best_cost:
                best_cost = cost
                best_loop = candidate[shift:] + candidate[:shift]

    return best_loop


def _quad_distortion_cost(p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> float:
    """Estimate geometric distortion cost for a quad (p0->p1 on dense, p3->p2 on sparse)."""
    # Bridge edge lengths (p0->p3 and p1->p2)
    d0 = (p3 - p0).length_squared
    d1 = (p2 - p1).length_squared

    # 1. Direction consistency: edges (p0->p1) and (p3->p2) must travel in the same direction
    e_dense = p1 - p0
    e_sparse = p2 - p3
    dir_pen = 0.0
    if e_dense.length > 1e-9 and e_sparse.length > 1e-9:
        cos_dir = e_dense.normalized().dot(e_sparse.normalized())
        if cos_dir < 0.0:
            dir_pen = 200.0 * (1.0 - cos_dir)  # Strong bowtie penalty
        elif cos_dir < 0.4:
            dir_pen = 15.0 * (0.4 - cos_dir)

    # 2. Normal consistency of the two split triangles (p0,p1,p2) and (p0,p2,p3)
    n1 = (p1 - p0).cross(p2 - p0)
    n2 = (p2 - p0).cross(p3 - p0)
    twist_pen = 0.0
    if n1.length > 1e-9 and n2.length > 1e-9:
        dot = n1.normalized().dot(n2.normalized())
        if dot < 0.0:
            twist_pen = 100.0 * (1.0 - dot)
        elif dot < 0.5:
            twist_pen = 10.0 * (0.5 - dot)

    # 3. Minimum interior angle penalty (avoid sliver/collapsed quads)
    pts = (p0, p1, p2, p3)
    min_ang = math.pi
    for k in range(4):
        va = pts[(k - 1) % 4] - pts[k]
        vb = pts[(k + 1) % 4] - pts[k]
        if va.length > 1e-9 and vb.length > 1e-9:
            ang = va.angle(vb)
            if ang < min_ang:
                min_ang = ang
        else:
            min_ang = 0.0
    ang_pen = 0.0
    min_deg = math.degrees(min_ang)
    if min_deg < 25.0:
        ang_pen = ((25.0 - min_deg) / 25.0) ** 2 * 20.0

    return d0 + d1 + dir_pen + twist_pen + ang_pen


_ADJ_3TO1_PENALTY = 50.0


def _build_3to1_cost_tables(
    loop_sparse: list[bmesh.types.BMVert],
    loop_dense: list[bmesh.types.BMVert],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-(sparse edge i, dense start vertex k) transition costs.

    A transition's cost depends only on the sparse edge index and the *absolute* dense start
    index, never on the cyclic shift being tried, so these tables are shared by every shift.
    Returns (c1, t3, a3, b3), each of shape (m, n):
      c1: cost of a 1-to-1 quad on dense edge k
      t3: 0.35 * summed distortion of the four quads of a 3-to-1 block starting at dense k
      a3, b3: the two bridge-length terms of that 3-to-1 block (kept separate so the DP adds
              them in the same order as the reference scalar formulation)
    """
    m = len(loop_sparse)
    n = len(loop_dense)
    c1 = np.empty((m, n))
    t3 = np.full((m, n), np.inf)
    a3 = np.zeros((m, n))
    b3 = np.zeros((m, n))
    dense_co = [v.co for v in loop_dense]
    for i in range(m):
        s0 = loop_sparse[i].co
        s1 = loop_sparse[(i + 1) % m].co
        for k in range(n):
            d0 = dense_co[k]
            d1 = dense_co[(k + 1) % n]
            c1[i, k] = _quad_distortion_cost(d0, d1, s1, s0)
            if n >= 3:
                d2 = dense_co[(k + 2) % n]
                d3 = dense_co[(k + 3) % n]
                p = d1.lerp(s0, 0.5)
                q = d2.lerp(s1, 0.5)
                t3[i, k] = 0.35 * (
                    _quad_distortion_cost(d0, d1, p, s0)
                    + _quad_distortion_cost(d1, d2, q, p)
                    + _quad_distortion_cost(d2, d3, s1, q)
                    + _quad_distortion_cost(p, q, s1, s0)
                )
                a3[i, k] = (d0 - s0).length_squared * 0.25
                b3[i, k] = (d3 - s1).length_squared * 0.25
    return c1, t3, a3, b3


def _solve_3to1_dp_batched(
    tables: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    shifts: np.ndarray,
    want_steps: bool = False,
) -> tuple[np.ndarray, list[int] | None]:
    """Solve the 1-to-1 / 3-to-1 matching DP for many cyclic shifts of the dense loop at once.

    dp[s, j, prev_was_3] = min cost matching sparse[0..i) to shifted dense[0..j) for shift s.
    Consecutive 3-to-1 blocks are penalized to prevent valence-6+ poles on the collar rim.
    Ties resolve exactly like the scalar reference (1-to-1 parent preferred from prev3=0,
    end state 0 preferred), so results are identical to the original per-shift solver.

    Returns (total_cost per shift [inf when infeasible], steps for shifts[0] if want_steps).
    """
    c1, t3, a3, b3 = tables
    m, n = c1.shape
    n_sh = len(shifts)
    j1 = np.arange(n)          # 1-to-1 starts: j + 1 <= n
    j3 = np.arange(max(n - 2, 0))  # 3-to-1 starts: j + 3 <= n
    idx1 = (shifts[:, None] + j1[None, :]) % n
    idx3 = (shifts[:, None] + j3[None, :]) % n

    dp0 = np.full((n_sh, n + 1), np.inf)
    dp1 = np.full((n_sh, n + 1), np.inf)
    dp0[:, 0] = 0.0
    parents: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(m):
        nd0 = np.full_like(dp0, np.inf)
        nd1 = np.full_like(dp1, np.inf)

        # Transition 1 (1-to-1) -> state 0
        cost1 = c1[i][idx1]
        from0 = dp0[:, j1] + cost1
        from1 = dp1[:, j1] + cost1
        pick1_0 = from1 < from0
        nd0[:, j1 + 1] = np.where(pick1_0, from1, from0)

        # Transition 2 (3-to-1) -> state 1
        if len(j3):
            tt = t3[i][idx3]
            aa = a3[i][idx3]
            bb = b3[i][idx3]
            from0 = (((dp0[:, j3] + 0.0) + tt) + aa) + bb
            from1 = (((dp1[:, j3] + _ADJ_3TO1_PENALTY) + tt) + aa) + bb
            pick1_1 = from1 < from0
            nd1[:, j3 + 3] = np.where(pick1_1, from1, from0)
        else:
            pick1_1 = np.zeros((n_sh, 0), dtype=bool)

        if want_steps:
            parents.append((pick1_0[0].copy(), pick1_1[0].copy()))
        dp0, dp1 = nd0, nd1

    end0 = dp0[:, n]
    end1 = dp1[:, n]
    totals = np.minimum(end0, end1)

    if not want_steps:
        return totals, None
    if not np.isfinite(totals[0]):
        return totals, [1] * m

    steps = [0] * m
    j = n
    st = 0 if end0[0] <= end1[0] else 1
    for i in range(m, 0, -1):
        pick1_0, pick1_1 = parents[i - 1]
        if st == 0:
            step = 1
            prev_st = 1 if pick1_0[j - 1] else 0
        else:
            step = 3
            prev_st = 1 if pick1_1[j - 3] else 0
        steps[i - 1] = step
        j -= step
        st = prev_st
    return totals, steps


def _project_and_relax_verts(
    internal_verts: list[bmesh.types.BMVert],
    bvh: BVHTree,
    iterations: int = 8,
) -> None:
    """Relax internal vertices via Laplacian smoothing with surface projection onto `bvh`."""
    if not internal_verts:
        return
    for _ in range(iterations):
        new_pos: dict[bmesh.types.BMVert, Vector] = {}
        for v in internal_verts:
            nbrs = [e.other_vert(v).co for e in v.link_edges]
            if not nbrs:
                continue
            avg = sum(nbrs, Vector()) / len(nbrs)
            blended = v.co.lerp(avg, 0.6)
            loc, _, _, _ = bvh.find_nearest(blended)
            new_pos[v] = loc.copy() if loc is not None else blended
        for v, pos in new_pos.items():
            v.co = pos


def _bridge_single_layer_3to1(
    bm: bmesh.types.BMesh,
    loop_sparse: list[bmesh.types.BMVert],
    loop_dense: list[bmesh.types.BMVert],
    side_id: int,
    bvh: BVHTree,
    layer_obj,
    layer_preserved,
) -> tuple[int, list[bmesh.types.BMVert]]:
    """Bridge `loop_sparse` (length m) and `loop_dense` (length n >= m, n <= 3m, n-m even) with all-quads."""
    m = len(loop_sparse)
    n = len(loop_dense)
    diff = n - m
    assert diff >= 0 and diff % 2 == 0 and n <= 3 * m

    # Find globally optimal orientation + cyclic shift + 3-to-1 placement via DP.
    # Every shift is solved at once from shared cost tables (first minimum wins, matching the
    # original orientation-then-shift scan order), then the winner is re-solved for its steps.
    all_shifts = np.arange(n)
    best_key: tuple[float, int, int] | None = None
    candidates = (list(loop_dense), list(reversed(loop_dense)))
    cand_tables = []
    for ci, candidate in enumerate(candidates):
        tables = _build_3to1_cost_tables(loop_sparse, candidate)
        cand_tables.append(tables)
        totals, _ = _solve_3to1_dp_batched(tables, all_shifts)
        s = int(np.argmin(totals))
        if best_key is None or totals[s] < best_key[0]:
            best_key = (float(totals[s]), ci, s)

    _, best_ci, best_shift = best_key
    if np.isfinite(best_key[0]):
        _, best_steps = _solve_3to1_dp_batched(
            cand_tables[best_ci], np.array([best_shift]), want_steps=True
        )
        candidate = candidates[best_ci]
        loop_dense = candidate[best_shift:] + candidate[:best_shift]
    else:
        best_steps = [1] * m
        loop_dense = list(loop_dense)
    faces_created = 0
    internal_verts: list[bmesh.types.BMVert] = []

    def _make_quad(v0, v1, v2, v3) -> None:
        nonlocal faces_created
        try:
            f = bm.faces.new((v0, v1, v2, v3))
            if layer_obj is not None:
                f[layer_obj] = side_id
            if layer_preserved is not None:
                f[layer_preserved] = 0
            faces_created += 1
        except ValueError:
            pass

    d_idx = 0
    for s_idx in range(m):
        s0 = loop_sparse[s_idx]
        s1 = loop_sparse[(s_idx + 1) % m]
        step = best_steps[s_idx]

        if step == 3:
            # 3-to-1 U-turn SubD reduction block (consumes 3 dense edges -> 1 sparse edge, 4 quads)
            d0 = loop_dense[d_idx % n]
            d1 = loop_dense[(d_idx + 1) % n]
            d2 = loop_dense[(d_idx + 2) % n]
            d3 = loop_dense[(d_idx + 3) % n]
            d_idx += 3

            p_raw = d1.co.lerp(s0.co, 0.5)
            q_raw = d2.co.lerp(s1.co, 0.5)
            p_loc, _, _, _ = bvh.find_nearest(p_raw)
            q_loc, _, _, _ = bvh.find_nearest(q_raw)
            vp = bm.verts.new(p_loc.copy() if p_loc is not None else p_raw)
            vq = bm.verts.new(q_loc.copy() if q_loc is not None else q_raw)
            internal_verts.extend((vp, vq))

            _make_quad(d0, d1, vp, s0)
            _make_quad(d1, d2, vq, vp)
            _make_quad(d2, d3, s1, vq)
            _make_quad(vp, vq, s1, s0)
        else:
            # Standard 1-to-1 quad bridge
            d0 = loop_dense[d_idx % n]
            d1 = loop_dense[(d_idx + 1) % n]
            d_idx += 1
            _make_quad(d0, d1, s1, s0)

    return faces_created, internal_verts


def _create_intermediate_ring(
    bm: bmesh.types.BMesh,
    loop_a: list[bmesh.types.BMVert],
    loop_b: list[bmesh.types.BMVert],
    target_count: int,
    bvh: BVHTree,
) -> list[bmesh.types.BMVert]:
    """Create an intermediate closed vertex ring midway between `loop_a` and `loop_b`."""
    m = len(loop_a)
    n = len(loop_b)
    ring: list[bmesh.types.BMVert] = []
    for k in range(target_count):
        t = k / target_count
        # Interpolate along loop_a
        fa = t * m
        ia0 = int(math.floor(fa)) % m
        ia1 = (ia0 + 1) % m
        pa = loop_a[ia0].co.lerp(loop_a[ia1].co, fa - math.floor(fa))

        # Interpolate along loop_b
        fb = t * n
        ib0 = int(math.floor(fb)) % n
        ib1 = (ib0 + 1) % n
        pb = loop_b[ib0].co.lerp(loop_b[ib1].co, fb - math.floor(fb))

        mid = pa.lerp(pb, 0.5)
        loc, _, _, _ = bvh.find_nearest(mid)
        ring.append(bm.verts.new(loc.copy() if loc is not None else mid))
    return ring


def fill_gap_pair(
    bm: bmesh.types.BMesh,
    gap: GapPair,
    bvh: BVHTree,
) -> tuple[int, int]:
    """Fill a single closed ring `GapPair` with an all-quad mesh."""
    if not gap.is_closed:
        return 0, 0
    inner = list(gap.inner_verts)
    outer = list(gap.outer_verts)
    if len(inner) < 3 or len(outer) < 3:
        return 0, 0

    # Parity check: if original mesh contained triangles causing odd parity, absorb 1 edge via a single tri
    extra_tri_faces = 0
    if (len(inner) + len(outer)) % 2 != 0:
        # Align first, then collapse one dense edge with a single triangle so remaining loops have even sum
        outer = _align_loop_orientation_and_shift(inner, outer)
        layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
        layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)
        if len(outer) >= len(inner) and len(outer) >= 4:
            # Make 1 triangle (outer[0], outer[1], inner[0]) and skip outer[0]->outer[1] via outer[1]
            # Better: insert midpoint on outer[0]->outer[1] is impossible without splitting untouched face;
            # instead close (outer[0], outer[1], inner[0]) and shift outer start by 1?
            # Wait: a triangle (o0, o1, i0) replaces edge (o0,o1) with edge (i0,o1), reducing outer count by 1?
            # Actually (o0, o1, i0) leaves edge (o1, i0) which connects to next quad (o1, o2, i1, i0) -> seamless!
            # So we simply pass outer_reduced = [o1, o2, ..., o_{n-1}, o0] with first quad using (o1,o2,i1,i0)...
            # Simpler & watertight: create tri (outer[0], outer[1], inner[0]) and bridge inner[0..m-1] with [outer[1]..outer[n-1]]?
            # That would leave (outer[n-1], outer[0], inner[0]) open! So instead we handle odd parity by
            # adding a single 2-to-1 triangle step inside a dedicated fallback bridge.
            pass

    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)

    # Align outer loop orientation and cyclic start index with inner loop
    outer = _align_loop_orientation_and_shift(inner, outer)

    # Odd-parity watertight fallback: if (m + n) is odd (e.g. non-quad input mesh),
    # bridge 1 dense edge as a single triangle and bridge the remaining (even-parity) chain with quads.
    if (len(inner) + len(outer)) % 2 != 0:
        swapped = len(inner) > len(outer)
        sparse_loop = outer if swapped else inner
        dense_loop = inner if swapped else outer
        m, n = len(sparse_loop), len(dense_loop)
        faces_created = 0
        # Consume dense[0..1] into a triangle with sparse[0], then match dense[1..n] (n-1 edges) to sparse[0..m] (m edges)
        try:
            f = bm.faces.new((dense_loop[0], dense_loop[1], sparse_loop[0]))
            if layer_obj is not None:
                f[layer_obj] = gap.side_id
            if layer_preserved is not None:
                f[layer_preserved] = 0
            faces_created += 1
        except ValueError:
            pass
        # Bridge remaining open chain: dense_rem has n-1 edges, sparse has m edges (sum is now even)
        d_rem = dense_loop[1:] + [dense_loop[0]]
        # Simple greedy proportional quad/tri strip for the remaining chain (guaranteed watertight)
        di = 0
        si = 0
        while si < m:
            s0 = sparse_loop[si]
            s1 = sparse_loop[(si + 1) % m]
            rem_s = m - si
            rem_d = (n - 1) - di
            if rem_d >= rem_s + 2 and di + 3 <= n - 1:
                # 3-to-1 quad block
                d0, d1, d2, d3 = d_rem[di], d_rem[di + 1], d_rem[di + 2], d_rem[di + 3]
                di += 3
                p_raw = d1.co.lerp(s0.co, 0.5)
                q_raw = d2.co.lerp(s1.co, 0.5)
                vp = bm.verts.new(p_raw)
                vq = bm.verts.new(q_raw)
                for quad_v in ((d0, d1, vp, s0), (d1, d2, vq, vp), (d2, d3, s1, vq), (vp, vq, s1, s0)):
                    try:
                        f = bm.faces.new(quad_v)
                        if layer_obj is not None:
                            f[layer_obj] = gap.side_id
                        if layer_preserved is not None:
                            f[layer_preserved] = 0
                        faces_created += 1
                    except ValueError:
                        pass
            elif rem_d >= 1:
                d0, d1 = d_rem[di], d_rem[di + 1]
                di += 1
                try:
                    f = bm.faces.new((d0, d1, s1, s0))
                    if layer_obj is not None:
                        f[layer_obj] = gap.side_id
                    if layer_preserved is not None:
                        f[layer_preserved] = 0
                    faces_created += 1
                except ValueError:
                    pass
            else:
                # Close remaining sparse edge to last dense vert
                try:
                    f = bm.faces.new((d_rem[di], s1, s0))
                    if layer_obj is not None:
                        f[layer_obj] = gap.side_id
                    if layer_preserved is not None:
                        f[layer_preserved] = 0
                    faces_created += 1
                except ValueError:
                    pass
            si += 1
        return faces_created, 0

    # Build multi-layer progression if density ratio exceeds 3:1
    # Always step from smaller loop to larger loop with ratio <= 3 per layer
    swapped = len(inner) > len(outer)
    sparse_loop = outer if swapped else inner
    dense_loop = inner if swapped else outer

    layers: list[list[bmesh.types.BMVert]] = [sparse_loop]
    all_internal: list[bmesh.types.BMVert] = []

    curr_loop = sparse_loop
    while len(dense_loop) > 3 * len(curr_loop):
        # Choose an intermediate even count <= 3 * len(curr_loop)
        mid_cnt = min(3 * len(curr_loop) - 2, (len(curr_loop) + len(dense_loop)) // 2)
        if mid_cnt % 2 != len(curr_loop) % 2:
            mid_cnt += 1
        mid_cnt = max(len(curr_loop) + 2, mid_cnt)
        mid_ring = _create_intermediate_ring(bm, curr_loop, dense_loop, mid_cnt, bvh)
        all_internal.extend(mid_ring)
        layers.append(mid_ring)
        curr_loop = mid_ring

    layers.append(dense_loop)

    total_faces = 0
    for i in range(len(layers) - 1):
        f_cnt, int_v = _bridge_single_layer_3to1(
            bm=bm,
            loop_sparse=layers[i],
            loop_dense=layers[i + 1],
            side_id=gap.side_id,
            bvh=bvh,
            layer_obj=layer_obj,
            layer_preserved=layer_preserved,
        )
        total_faces += f_cnt
        all_internal.extend(int_v)

    _project_and_relax_verts(all_internal, bvh=bvh, iterations=6)
    return total_faces, len(all_internal)


def fill_cap_loop(
    bm: bmesh.types.BMesh,
    cap: CapLoop,
    bvh: BVHTree,
) -> tuple[int, int]:
    """Fill an even-sided closed boundary loop with a Coons-interpolated a x b all-quad grid."""
    verts = list(cap.verts)
    m = len(verts)
    if m < 4 or m % 2 != 0:
        return 0, 0

    layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
    layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)

    if m == 4:
        try:
            f = bm.faces.new((verts[0], verts[1], verts[2], verts[3]))
            if layer_obj is not None:
                f[layer_obj] = cap.side_id
            if layer_preserved is not None:
                f[layer_preserved] = 0
            return 1, 0
        except ValueError:
            return 0, 0

    # Decompose m = 2*(a + b) with a <= b, a >= 1, b >= 2
    half = m // 2
    a = max(1, half // 2)
    b = half - a

    # Map perimeter vertices to (a+1) x (b+1) grid boundary in counter-clockwise order
    grid: list[list[bmesh.types.BMVert | None]] = [[None] * (b + 1) for _ in range(a + 1)]
    idx = 0
    for j in range(b):
        grid[0][j] = verts[idx]; idx += 1
    for i in range(a):
        grid[i][b] = verts[idx]; idx += 1
    for j in range(b, 0, -1):
        grid[a][j] = verts[idx]; idx += 1
    for i in range(a, 0, -1):
        grid[i][0] = verts[idx]; idx += 1

    # Coons patch bilinear interpolation for interior grid vertices + surface projection
    internal: list[bmesh.types.BMVert] = []
    c00 = grid[0][0].co
    c0b = grid[0][b].co
    ca0 = grid[a][0].co
    cab = grid[a][b].co
    for i in range(1, a):
        u = i / a
        for j in range(1, b):
            v = j / b
            p_u = grid[0][j].co.lerp(grid[a][j].co, u)
            p_v = grid[i][0].co.lerp(grid[i][b].co, v)
            p_bilin = (
                (1 - u) * (1 - v) * c00
                + (1 - u) * v * c0b
                + u * (1 - v) * ca0
                + u * v * cab
            )
            pt = p_u + p_v - p_bilin
            loc, _, _, _ = bvh.find_nearest(pt)
            nv = bm.verts.new(loc.copy() if loc is not None else pt)
            grid[i][j] = nv
            internal.append(nv)

    faces_created = 0
    for i in range(a):
        for j in range(b):
            try:
                f = bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))
                if layer_obj is not None:
                    f[layer_obj] = cap.side_id
                if layer_preserved is not None:
                    f[layer_preserved] = 0
                faces_created += 1
            except ValueError:
                pass

    _project_and_relax_verts(internal, bvh=bvh, iterations=8)
    return faces_created, len(internal)


def quadrangulate_all_gaps(
    bm: bmesh.types.BMesh,
    gap_pairs: list[GapPair],
    bvh_a: BVHTree,
    bvh_b: BVHTree,
    cap_loops: list[CapLoop] | None = None,
) -> LocalRemeshResult:
    """Execute Stage 3 across all detected gap pairs and cap loops, and make normals consistent."""
    res = LocalRemeshResult()
    for gap in gap_pairs:
        bvh = bvh_a if gap.side_id == 1 else bvh_b
        f_cnt, v_cnt = fill_gap_pair(bm=bm, gap=gap, bvh=bvh)
        if f_cnt > 0:
            res.faces_created += f_cnt
            res.internal_verts_created += v_cnt
            res.gaps_filled += 1
        else:
            res.gaps_skipped += 1

    if cap_loops:
        for cap in cap_loops:
            bvh = bvh_a if cap.side_id == 1 else bvh_b
            f_cnt, v_cnt = fill_cap_loop(bm=bm, cap=cap, bvh=bvh)
            res.faces_created += f_cnt
            res.internal_verts_created += v_cnt

    # Ensure consistent outward winding across stitched regions
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    return res
