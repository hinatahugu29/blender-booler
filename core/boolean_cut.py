"""Stage 1: Exact/Manifold Boolean execution, original-face tracking, and intersection loop extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import bmesh
import bpy


BooleanOp = Literal["UNION", "DIFFERENCE", "INTERSECT"]
BooleanSolver = Literal["EXACT", "MANIFOLD", "FLOAT"]

ATTR_ORIG_OBJ = "booler_orig_obj"
ATTR_ORIG_FACE = "booler_orig_face"
ATTR_PRESERVED = "booler_preserved"


@dataclass
class IntersectionLoop:
    """An ordered chain of intersection vertices and edges."""
    verts: list[bmesh.types.BMVert]
    edges: list[bmesh.types.BMEdge]
    is_closed: bool


@dataclass
class BooleanCutResult:
    """Output of Stage 1 Boolean cut."""
    bm: bmesh.types.BMesh
    inter_edges: list[bmesh.types.BMEdge]
    inter_loops: list[IntersectionLoop]
    untouched_face_count: int
    total_face_count: int


def _tag_mesh_attributes(mesh: bpy.types.Mesh, obj_id: int) -> None:
    """Tag every polygon of `mesh` with its source object ID and polygon index."""
    for name in (ATTR_ORIG_OBJ, ATTR_ORIG_FACE):
        if name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[name])

    n = len(mesh.polygons)
    attr_obj = mesh.attributes.new(ATTR_ORIG_OBJ, "INT", "FACE")
    attr_face = mesh.attributes.new(ATTR_ORIG_FACE, "INT", "FACE")
    if n > 0:
        attr_obj.data.foreach_set("value", [obj_id] * n)
        attr_face.data.foreach_set("value", list(range(n)))


def extract_intersection_loops(
    inter_edges: list[bmesh.types.BMEdge],
) -> list[IntersectionLoop]:
    """Order an unordered list of intersection BMEdges into closed/open chains."""
    if not inter_edges:
        return []

    edge_set = set(inter_edges)
    adj: dict[bmesh.types.BMVert, list[bmesh.types.BMEdge]] = {}
    for e in inter_edges:
        adj.setdefault(e.verts[0], []).append(e)
        adj.setdefault(e.verts[1], []).append(e)

    visited_edges: set[bmesh.types.BMEdge] = set()
    loops: list[IntersectionLoop] = []

    def _trace(start_v: bmesh.types.BMVert, start_e: bmesh.types.BMEdge) -> IntersectionLoop:
        verts = [start_v]
        edges = [start_e]
        visited_edges.add(start_e)
        curr_v = start_e.other_vert(start_v)
        verts.append(curr_v)

        while True:
            # Stop if we returned to start (closed loop)
            if curr_v == start_v:
                return IntersectionLoop(verts=verts[:-1], edges=edges, is_closed=True)

            # Stop at junctions (degree != 2) or endpoints (degree == 1)
            inc = [e for e in adj[curr_v] if e not in visited_edges]
            if not inc:
                return IntersectionLoop(verts=verts, edges=edges, is_closed=False)
            if len(adj[curr_v]) != 2:
                return IntersectionLoop(verts=verts, edges=edges, is_closed=False)

            next_e = inc[0]
            visited_edges.add(next_e)
            edges.append(next_e)
            curr_v = next_e.other_vert(curr_v)
            verts.append(curr_v)

    # 1. Start from non-degree-2 vertices first (endpoints / T-junctions)
    for v, inc_edges in adj.items():
        if len(inc_edges) != 2:
            for e in inc_edges:
                if e not in visited_edges:
                    loops.append(_trace(v, e))

    # 2. Trace remaining pure closed loops (all degree-2)
    for e in inter_edges:
        if e not in visited_edges:
            loops.append(_trace(e.verts[0], e))

    return loops


def execute_boolean_cut(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    operation: BooleanOp = "UNION",
    solver: BooleanSolver = "EXACT",
) -> BooleanCutResult:
    """Run non-destructive Boolean evaluation and return tagged BMesh + intersection loops."""
    tmp_a = obj_a.copy()
    tmp_a.data = obj_a.data.copy()
    tmp_b = obj_b.copy()
    tmp_b.data = obj_b.data.copy()

    bpy.context.collection.objects.link(tmp_a)
    bpy.context.collection.objects.link(tmp_b)
    # Ensure temporary copies are visible to depsgraph even when non-destructive sources are hidden
    for tmp in (tmp_a, tmp_b):
        tmp.hide_viewport = False
        tmp.hide_set(False)

    try:
        _tag_mesh_attributes(tmp_a.data, obj_id=1)
        _tag_mesh_attributes(tmp_b.data, obj_id=2)

        mod = tmp_a.modifiers.new("BoolerCut", "BOOLEAN")
        mod.operation = operation
        mod.solver = solver
        mod.object = tmp_b

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = tmp_a.evaluated_get(depsgraph)
        eval_me = eval_obj.to_mesh()

        bm = bmesh.new()
        try:
            bm.from_mesh(eval_me)
        finally:
            eval_obj.to_mesh_clear()

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        layer_obj = bm.faces.layers.int.get(ATTR_ORIG_OBJ)
        layer_preserved = bm.faces.layers.int.get(ATTR_PRESERVED)
        if layer_preserved is None:
            layer_preserved = bm.faces.layers.int.new(ATTR_PRESERVED)

        inter_edges: list[bmesh.types.BMEdge] = []
        inter_verts: set[bmesh.types.BMVert] = set()
        if layer_obj is not None:
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    if e.link_faces[0][layer_obj] != e.link_faces[1][layer_obj]:
                        inter_edges.append(e)
                        inter_verts.add(e.verts[0])
                        inter_verts.add(e.verts[1])

        # Mark faces that do not touch any intersection vertex as preserved
        untouched = 0
        for f in bm.faces:
            if any(v in inter_verts for v in f.verts):
                f[layer_preserved] = 0
            else:
                f[layer_preserved] = 1
                untouched += 1

        inter_loops = extract_intersection_loops(inter_edges)

        return BooleanCutResult(
            bm=bm,
            inter_edges=inter_edges,
            inter_loops=inter_loops,
            untouched_face_count=untouched,
            total_face_count=len(bm.faces),
        )
    finally:
        for obj in (tmp_a, tmp_b):
            me = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(me, do_unlink=True)
