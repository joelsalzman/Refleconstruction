import bpy
import bmesh
import math
from mathutils import Matrix, Vector, Euler

# We can ignore z coords
# NEED (X,Y,Z) COORDS OF CENTER OF OBJECT
# NEED (X,Y,Z) COORDS OF MIRRORS
# NEED (X,Y,Z) COORDS OF THE REFLECTION
# THE REALSENSE IS ABOVE THE GROUND SO WE NEED THE BOUNDING VOLUME TO GO DOWN A BIT AS WELL

OBJ = (0, -0.08, -0.5)
MIRR = (-0.51, 0, -0.76)
REF = (-0.77, 0, -1.24)

box_width = 0.4  # x
box_height = 1  # y
box_depth = 0.4  # z

points = {
    "OBJ": Vector(OBJ),
    "MIRR": Vector(MIRR),
    "REF": Vector(REF),
}

FILEPATH = "/Users/nikh/Columbia/compimg_6732/CI-Project/data/parrot_test_5.ply"


def clear_scene():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)


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


def align_meshes(obj_mesh, ref_mesh, mirror_normal):
    """
    Aligns the ref_mesh to the obj_mesh based on a mirror plane defined by mirror_normal.

    Parameters:
    - obj_mesh: The target mesh object (Blender object)
    - ref_mesh: The reference mesh object to align (Blender object)
    - mirror_normal: A Vector representing the normal to the mirror plane
    """
    # Compute the required rotation
    mirror_normal.normalize()
    # the realsense reads out with y pointing up and z pointing forwards
    up_vector = Vector((0, 1, 0))

    # Calculate the rotation matrix 
    rotation_axis = up_vector.cross(mirror_normal)
    rotation_angle = up_vector.angle(mirror_normal)
    rotation_matrix = Matrix.Rotation(rotation_angle, 4, rotation_axis)

    # Correcting for the plane reflection: rotate 180 degrees around the mirror normal
    # Might need to invert here instead but not sure yet
    reflection_matrix = Matrix.Rotation(math.pi, 4, mirror_normal)

    # Apply the computed rotation to the REF mesh
    ref_mesh.matrix_world = (
        Matrix.Translation(obj_mesh.location)  # Translate to OBJ mesh location
        @ rotation_matrix  # Align with mirror normal
        @ reflection_matrix  # Reflect across the mirror plane
        @ Matrix.Translation(
            -ref_mesh.location
        )  # Translate center to origin for rotation
    )


### RUN
clear_scene()
load_and_rotate_ply(FILEPATH)
original_obj = bpy.context.scene.objects["Imported_PLY_Object"]

# Create meshes around each point
for key, point in points.items():
    create_mesh_from_box(
        original_obj, point, box_width, box_height, box_depth, f"Mesh_{key}"
    )

obj_mesh = bpy.data.objects['Mesh_OBJ']
    ref_mesh = bpy.data.objects['Mesh_REF']
    mirror_normal = Vector((0,0,0)) - MIR

    align_meshes(obj_mesh, ref_mesh, mirror_normal)