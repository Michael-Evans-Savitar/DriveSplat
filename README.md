![pipeline diagram](assets/pipeline.jpg)
## DriveSplat: Decoupled Dynamic Scenario Representation with Partitioned Neural Gaussians for Driving Scene Reconstruction

# 📖 Overview
We introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors such as pedestrians and cyclists, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. 

# 👀 Demo
<table>
  <tr>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 006</p>
    </td>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 026</p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 090</p>
    </td>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 105</p>
    </td>
  </tr>
  <tr>
      <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 108</p>
    </td>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 134</p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 150</p>
    </td>
    <td align="center">
      <img src="assets/4月17日.gif" width="100%">
      <p align="center">Scene 181</p>
    </td>
  </tr>
</table>

