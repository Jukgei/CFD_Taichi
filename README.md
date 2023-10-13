## SPH in Taichi
<p align=center>
 <img src="https://github.com/Jukgei/CFD_Taichi/blob/main/demo/breaking_dam.gif" width="80%" height="80%"/>
</p>
The aim of this project is to learn and comprehend the 
Smoothed-particle hydrodynamics (SPH) method and its classical 
variants. Five classical SPH solvers and one classical rigid-fluid 
coupling method are implemented in [Taichi Lang [1]](https://github.com/taichi-dev/taichi). 
Due to the aforementioned motivation, our objective is to make the code as understandable as possible. Therefore, we strive to align the code formulas with those mentioned in the research papers.
As a result, we have limited the optimization of the code, which means that the simulator may not be as efficient as it could be.

## Prerequisites
- taichi
- numpy
- trimesh
- ffmpeg
- vulkan

## Feature
| Solver                                                              | Single-phase Flow | [Rigid-Fluid Coupling [7]](https://dl.acm.org/doi/abs/10.1145/2185520.2185558) |
|---------------------------------------------------------------------|-------------------|--------------------------------------------------------------------------------|
| [WCSPH [2]](https://dl.acm.org/doi/abs/10.5555/1272690.1272719)     | :o:               | :o:                                                                            |
| [PBF [3]](https://dl.acm.org/doi/abs/10.1145/2461912.2461984)       | :o:               | :x:                                                                            |
| [PCISPH [4]](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)    | :o:               | :o:                                                                            |
| [IISPH [5]](https://ieeexplore.ieee.org/abstract/document/6570475/) | :o:               | :o:                                                                            |
| [DFSPH [6]](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)     | :o:               | :o:                                                                            |
- Neighbor Searching based on grid
- Simple rigid simulator

## Usage
To be completed.


## Acknowledgement
The implementation is largely inspired by [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) and erizmr's [SPH_Taichi](https://github.com/erizmr/SPH_Taichi).

## Reference
1. [Taichi Lang](https://github.com/taichi-dev/taichi)
2. [WCSPH](https://dl.acm.org/doi/abs/10.5555/1272690.1272719)
3. [PBF](https://dl.acm.org/doi/abs/10.1145/2461912.2461984)
4. [PCISPH](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)
5. [IISPH](https://ieeexplore.ieee.org/abstract/document/6570475/)
6. [DFSPH](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)
7. [Rigid-Fluid Coupling](https://dl.acm.org/doi/abs/10.1145/2185520.2185558)