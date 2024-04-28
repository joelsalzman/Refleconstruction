import bpy

# Check if the PLY import/export add-on is registered
print("io_mesh_ply" in bpy.context.preferences.addons.keys())
