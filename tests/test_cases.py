"""Standard benchmark test cases for Booler."""

from __future__ import annotations

import math
from typing import Callable

import bmesh
import bpy
from mathutils import Matrix


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)


def _create_quad_cube(name: str, size: float = 2.0, cuts: int = 3, location=(0, 0, 0), rotation=(0, 0, 0)) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    if cuts > 0:
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=cuts, use_grid_fill=True)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = location
    obj.rotation_euler = rotation
    bpy.context.collection.objects.link(obj)
    return obj


def _create_quad_cylinder(
    name: str,
    radius: float = 0.6,
    depth: float = 2.4,
    segments: int = 16,
    rings: int = 6,
    location=(0, 0, 0),
    rotation=(0, 0, 0),
) -> bpy.types.Object:
    """Create an all-quad cylinder (grid-filled caps via subdivided cube or ring-capped)."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments,
        radius=radius,
        depth=depth,
        end_fill_type="NOTHING",
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    obj.name = name

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Add longitudinal rings
    side_edges = [e for e in bm.edges if abs((e.verts[0].co - e.verts[1].co).z) > 1e-4]
    if rings > 1:
        bmesh.ops.subdivide_edges(bm, edges=side_edges, cuts=rings - 1, use_grid_fill=True)

    # Grid fill top and bottom boundary loops using edit-mode fill_grid
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    for is_top in (True, False):
        for elem in bm.verts:
            elem.select = False
        for elem in bm.edges:
            elem.select = False
        for elem in bm.faces:
            elem.select = False
        for e in bm.edges:
            if e.is_boundary and ((e.verts[0].co.z > 0) == is_top):
                e.select = True
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.fill_grid()
        bm = bmesh.from_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def _create_quad_sphere(name: str, radius: float = 1.0, subdivisions: int = 3, location=(0, 0, 0)) -> bpy.types.Object:
    """Create an all-quad sphere by spherifying a subdivided cube."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2**subdivisions - 1, use_grid_fill=True)
    for v in bm.verts:
        v.co = v.co.normalized() * radius
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def case_1_box_box_union() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    _clear_scene()
    a = _create_quad_cube("BoxA", size=2.0, cuts=3, location=(0, 0, 0))
    b = _create_quad_cube("BoxB", size=1.4, cuts=2, location=(0.8, 0.3, 0.4), rotation=(0, 0, math.radians(20)))
    return a, b, "UNION"


def case_2_cube_cyl_diff() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    _clear_scene()
    a = _create_quad_cube("BaseCube", size=2.0, cuts=4, location=(0, 0, 0))
    b = _create_quad_cylinder("HoleCyl", radius=0.45, depth=3.0, segments=16, rings=6, location=(0.15, 0.1, 0))
    return a, b, "DIFFERENCE"


def case_3_cyl_cyl_t_union() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    _clear_scene()
    a = _create_quad_cylinder("MainPipe", radius=0.6, depth=2.6, segments=20, rings=8, location=(0, 0, 0))
    b = _create_quad_cylinder(
        "BranchPipe",
        radius=0.4,
        depth=1.8,
        segments=16,
        rings=6,
        location=(0.6, 0, 0.1),
        rotation=(0, math.radians(90), 0),
    )
    return a, b, "UNION"


def case_4_sphere_cyl_union() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    _clear_scene()
    a = _create_quad_sphere("QuadSphere", radius=1.0, subdivisions=3, location=(0, 0, 0))
    b = _create_quad_cylinder(
        "StubCyl",
        radius=0.38,
        depth=1.6,
        segments=16,
        rings=6,
        location=(0.7, 0.0, 0.2),
        rotation=(0, math.radians(75), 0),
    )
    return a, b, "UNION"


def case_5_box_cyl_intersect() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    _clear_scene()
    a = _create_quad_cube("BoxI", size=2.0, cuts=4, location=(0, 0, 0))
    b = _create_quad_cylinder("CylI", radius=1.1, depth=3.0, segments=24, rings=6, location=(0.2, 0.2, 0))
    return a, b, "INTERSECT"


def case_6_density_mismatch() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    """Coarse cube (cuts=1) intersected by dense cylinder (segments=32)."""
    _clear_scene()
    a = _create_quad_cube("CoarseCube", size=2.2, cuts=1, location=(0, 0, 0))
    b = _create_quad_cylinder("DenseCyl", radius=0.45, depth=3.0, segments=32, rings=8, location=(0.1, 0.1, 0))
    return a, b, "DIFFERENCE"


def case_7_non_uniform_scale() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    """Unapplied non-uniform scale & rotation on both meshes."""
    _clear_scene()
    a = _create_quad_cube("ScaledBox", size=2.0, cuts=4, location=(0.3, -0.2, 0.1), rotation=(0.2, 0.3, 0.5))
    a.scale = (1.5, 0.8, 1.2)
    b = _create_quad_cylinder(
        "ScaledCyl",
        radius=0.4,
        depth=2.4,
        segments=16,
        rings=6,
        location=(0.5, 0.0, 0.3),
        rotation=(0.7, 0.4, -0.3),
    )
    b.scale = (1.0, 1.4, 1.0)
    bpy.context.view_layer.update()
    return a, b, "UNION"


def case_8_chained_boolean() -> tuple[bpy.types.Object, bpy.types.Object, str]:
    """Chained Booler: run Union(Box, Cyl) first, then Difference a second Cyl on the Booler result."""
    from Booler.core.pipeline import BoolerConfig, run_booler_pipeline

    _clear_scene()
    base = _create_quad_cube("ChainBase", size=2.0, cuts=4, location=(0, 0, 0))
    boss = _create_quad_cylinder(
        "ChainBoss",
        radius=0.5,
        depth=1.6,
        segments=16,
        rings=6,
        location=(0.0, 0.0, 0.9),
    )
    step1, _ = run_booler_pipeline(base, boss, BoolerConfig(operation="UNION"), result_name="ChainStep1")
    hole = _create_quad_cylinder(
        "ChainHole",
        radius=0.3,
        depth=3.0,
        segments=16,
        rings=6,
        location=(0.6, 0.0, 0.0),
        rotation=(0, math.radians(90), 0),
    )
    return step1, hole, "DIFFERENCE"


ALL_CASES: list[tuple[str, Callable[[], tuple[bpy.types.Object, bpy.types.Object, str]]]] = [
    ("Case1_BoxBox_Union", case_1_box_box_union),
    ("Case2_CubeCyl_Diff", case_2_cube_cyl_diff),
    ("Case3_CylCyl_T_Union", case_3_cyl_cyl_t_union),
    ("Case4_SphereCyl_Union", case_4_sphere_cyl_union),
    ("Case5_BoxCyl_Intersect", case_5_box_cyl_intersect),
    ("Case6_DensityMismatch_Diff", case_6_density_mismatch),
    ("Case7_NonUniformScale_Union", case_7_non_uniform_scale),
    ("Case8_ChainedBooler_Diff", case_8_chained_boolean),
]
