import bpy

def torch_to_blender(vertices):

    # Convert the PyTorch tensor to a NumPy array
    vertices_np = vertices.numpy()
    
    # Create a new mesh and object
    mesh = bpy.data.meshes.new(name='VertexMesh')
    obj = bpy.data.objects.new(name='VertexObject', object_data=mesh)
    
    # Link the object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    
    # Update the mesh with the vertices
    mesh.from_pydata(vertices_np, [], [])
    mesh.update()
    
    # Set the object as the active object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)