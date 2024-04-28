import numpy as np
import bpy
import bmesh
import math
from mathutils import Matrix, Vector, Euler

# from transform import reflect_points

# We can ignore z coords
# NEED (X,Y,Z) COORDS OF CENTER OF OBJECT
# NEED (X,Y,Z) COORDS OF MIRRORS
# NEED (X,Y,Z) COORDS OF THE REFLECTION
# THE REALSENSE IS ABOVE THE GROUND SO WE NEED THE BOUNDING VOLUME TO GO DOWN A BIT AS WELL

OBJ = Vector((0, -0.08, -0.5))
MIRR = Vector((-0.51, 0, -0.76))
REF = Vector((-0.77, 0, -1.24))

box_width = 0.4  # x
box_height = 1  # y
box_depth = 0.4  # z

points = {
    "OBJ": OBJ,
    "MIRR": MIRR,
    "REF": REF,
}

# FILEPATH = "/Users/nikh/Columbia/compimg_6732/CI-Project/data/cup_test_1.ply"
FILEPATH = "/Users/nikh/Columbia/compimg_6732/CI-Project/data/parrot_test_5.ply"

bpy.ops.preferences.addon_enable(module="io_mesh_ply")


#### UTIL FUNCTIONS TODO refactor these out ####
def save_mesh_as_obj(mesh_object, file_path):
    """
    Saves the specified mesh object to an OBJ file at the given file path.
    """
    bpy.ops.object.select_all(action="DESELECT")
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)


def fit_plane_to_vertices(obj_name):
    """Fits a plane to the vertices of the given mesh object and returns the plane normal."""
    obj = bpy.data.objects[obj_name]
    mesh = obj.data

    coords = [obj.matrix_world @ v.co for v in mesh.vertices]
    coords = np.array([[v.x, v.y, v.z] for v in coords])
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid

    H = np.dot(coords_centered.T, coords_centered)
    U, S, V = np.linalg.svd(H)

    plane_normal = Vector(V[-1])
    tangent1 = Vector(V[0])
    tangent2 = Vector(V[1])

    return plane_normal, tangent1, tangent2


def clear_scene():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)


def find_closest_vertices(obj_a, obj_b, height_axis="z", height_tolerance=0.1):
    """
    Find the closest vertices between two objects considering only vertices that are within a certain height tolerance.

    Parameters:
    - obj_a, obj_b: The mesh objects to compare.
    - height_axis: The axis to use as height ('x', 'y', or 'z').
    - height_tolerance: The maximum difference in height coordinates to consider vertices for comparison.
    """
    mesh_a = obj_a.data
    mesh_b = obj_b.data

    # I know that realsense loads in with y as default, but it feels wrong
    axis_index = {"x": 0, "y": 1, "z": 2}.get(height_axis, 2)

    min_distance = float("inf")
    closest_pair = (None, None)

    edge_a = {v for e in mesh_a.edges for v in e.vertices}
    edge_b = {v for e in mesh_b.edges for v in e.vertices}

    # We 'slice' both meshes TODO only compare edge vertices
    for idx_a in edge_a:

        vert_a = mesh_a.vertices[idx_a]
        world_vert_a = obj_a.matrix_world @ vert_a.co

        for idx_b in edge_b:
            vert_b = mesh_b.vertices[idx_b]
            world_vert_b = obj_b.matrix_world @ vert_b.co

            if (
                abs(world_vert_a[axis_index] - world_vert_b[axis_index])
                <= height_tolerance
            ):
                distance = (world_vert_a - world_vert_b).length
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (world_vert_a, world_vert_b)
    return closest_pair


def calculate_translation(closest_vertices):
    vert_a, vert_b = closest_vertices
    # Calculate translation vector (from vert_b to vert_a)
    translation_vector = vert_a - vert_b
    return translation_vector


#### UTIL FUNCTIONS TODO refactor these out ####


def load_and_rotate_ply(filepath):
    clear_scene()

    bpy.ops.import_mesh.ply(filepath=filepath)

    obj = bpy.context.selected_objects[0]
    obj.name = "Imported_PLY_Object"
    # obj.location = (0, 0, 0)

    # # first rotate x by 90 deg then rotate by 180 around z
    # obj.rotation_euler[0] += math.radians(90)
    # obj.rotation_euler[2] += math.radians(180)


def create_mesh_from_box(base_mesh, center, width, height, depth, name):
    """
    Creates a new mesh from all mesh elements (vertices, edges, faces)
    The function draws a bounding rectangle from the xyz coordinate upwards for each object
    """
    bm = bmesh.new()
    bm.from_mesh(base_mesh.data)
    new_mesh = bpy.data.meshes.new(name)
    new_bm = bmesh.new()

    xmin, xmax = center.x - width / 2, center.x + width / 2
    ymin, ymax = center.y - height / 2, center.y + height / 2
    zmin, zmax = center.z - depth / 2, center.z + depth / 2

    vert_map = {}

    for vert in bm.verts:
        if (
            xmin <= vert.co.x <= xmax
            and ymin <= vert.co.y <= ymax
            and zmin <= vert.co.z <= zmax
        ):
            new_vert = new_bm.verts.new(vert.co)
            vert_map[vert.index] = new_vert

    # Add edges and faces whose vertices are all included in the new mesh
    for edge in bm.edges:
        if edge.verts[0].index in vert_map and edge.verts[1].index in vert_map:
            new_bm.edges.new(
                (vert_map[edge.verts[0].index], vert_map[edge.verts[1].index])
            )

    for face in bm.faces:
        if all(v.index in vert_map for v in face.verts):
            new_bm.faces.new([vert_map[v.index] for v in face.verts])

    new_bm.to_mesh(new_mesh)
    new_bm.free()

    # Add the new mesh as an object into the scene, located at the original center
    new_obj = bpy.data.objects.new(name, new_mesh)
    bpy.context.collection.objects.link(new_obj)


def rotate_mesh_in_edit_mode(obj_name):
    # Ensure the correct object is active and selected
    bpy.ops.object.select_all(action="DESELECT")  # Deselect all objects
    obj = bpy.data.objects[obj_name]  # Get the object by name
    bpy.context.view_layer.objects.active = obj  # Make it the active object
    obj.select_set(True)  # Select the object

    # Toggle to Edit Mode
    bpy.ops.object.mode_set(mode="EDIT")

    # Change shading to WIREFRAME to visually inspect the mesh during editing
    # This is usually not necessary for the script to function but is useful for debugging
    area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
    area.spaces[0].shading.type = "WIREFRAME"

    # Select all geometry in the mesh
    bpy.ops.mesh.select_all(action="SELECT")

    # Apply rotation
    bpy.ops.transform.rotate(
        value=3.14159,  # Radians, equivalent to 180 degrees
        orient_axis="Y",
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        constraint_axis=(False, True, False),  # Constrain to Y-axis
        use_proportional_edit=False,
    )

    # Toggle back to Object Mode
    bpy.ops.object.mode_set(mode="OBJECT")


def translate_using_mirror_in_edit_mode(obj_name, mirror_normal, mirror_point):
    """
    Translates an object in edit mode so that it aligns with its reflection across a given mirror plane.

    Parameters:
    - obj_name: Name of the object to translate.
    - mirror_normal: A Vector representing the normal of the mirror plane.
    - mirror_point: A Vector representing a point on the mirror plane.
    """
    # Ensure correct object is selected and active
    bpy.ops.object.select_all(action="DESELECT")
    obj = bpy.data.objects[obj_name]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Switch to edit mode
    bpy.ops.object.mode_set(mode="EDIT")

    # Load the mesh data into bmesh for edit
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Calculate the reflection matrix based on the mirror plane
    d = -mirror_normal.dot(mirror_point)
    reflection_matrix = Matrix(
        [
            [
                1 - 2 * mirror_normal.x**2,
                -2 * mirror_normal.x * mirror_normal.y,
                -2 * mirror_normal.x * mirror_normal.z,
            ],
            [
                -2 * mirror_normal.y * mirror_normal.x,
                1 - 2 * mirror_normal.y**2,
                -2 * mirror_normal.y * mirror_normal.z,
            ],
            [
                -2 * mirror_normal.z * mirror_normal.x,
                -2 * mirror_normal.z * mirror_normal.y,
                1 - 2 * mirror_normal.z**2,
            ],
        ]
    )

    # We use the center of the geometry for simplicity
    center_of_geometry = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
    reflected_position = reflection_matrix @ center_of_geometry + mirror_normal * (
        2 * d
    )
    translation_vector = reflected_position - center_of_geometry

    # Translate all vertices by the calculated vector
    for v in bm.verts:
        v.co += translation_vector

    # Update the mesh and switch back to object mode
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")


### RUN
# clear_scene()
load_and_rotate_ply(FILEPATH)
# original_obj = bpy.context.scene.objects["Imported_PLY_Object"]

# # Create meshes around each point
# for key, point in points.items():
#     create_mesh_from_box(
#         original_obj, point, box_width, box_height, box_depth, f"Mesh_{key}"
#     )


# obj_mesh = bpy.data.objects["Mesh_OBJ"]
# ref_mesh = bpy.data.objects["Mesh_REF"]
# mirr_mesh = bpy.data.objects["Mesh_MIRR"]


# save_mesh_as_obj(obj_mesh, "Mesh_OBJ.obj")
# save_mesh_as_obj(ref_mesh, "Mesh_REF.obj")
# save_mesh_as_obj(mirr_mesh, "Mesh_MIRR.obj")

# mirror_normal = OBJ - MIRR

# # MAP ONTO
# # rotate_mesh_in_edit_mode("Mesh_REF")
# mirror, _, normal = fit_plane_to_vertices("Mesh_MIRR")
# mn = normal.normalized()
# print(mirror)
# # translate_using_mirror_in_edit_mode("Mesh_REF", mirror_normal=mn, mirror_point=MIRR)
