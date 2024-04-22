# CI Project - Better 3D reconstruction using stereo through mirrors

Computational Imaging Project

## Methodology

We use the Realsense and mirrors to try and get a better 3d reconstruction of object.

### 1. Get an rgb and depth image from the realsense
<div style="width: 100%; text-align: center;">
  <table style="margin: auto;">
    <tr>
      <td><img src="media/parrot_test_5_Color.png" alt="RGB Image" width="500"/></td>
      <td><img src="media/parrot_test_5_D_Depth.png" alt="Depth Image" width="500"/></td>
    </tr>
  </table>
</div>


### 2. Load the mesh and segment out the object and reflection
We use blender for this step due to easy visualisation and python integration. A faster system might use opengl in cpp. We use the positions of the object and the mirror to determine the reflection's position. In code, we draw bounding volumes around each mesh and crop out two sub meshes for the object, and it's reflection.

<div style="width: 100%; text-align: center;">
  <table style="margin: auto;">
    <tr>
      <td><img src="media/mesh.png" alt="RGB Image" width="500"/></td>
      <td><img src="media/seg.png" alt="Depth Image" width="500"/></td>
    </tr>
  </table>
</div>

### 3. Transform and merge reflection onto object


### 4. Map textures