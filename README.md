# MorE3D -- Subsampling 

Subsampling files created in the framework of MSCA MorE3D project.    

- main.py - main running code. For now it runs the entire saliency computation process using OPALS. 

- DM_saliency.py - use of pyDM to compute the saliency. Two processors are defined: 
    
    - curvature computation
    - dk and dn computation 
- BallTreePointSet.py - a class for the ball tree data structure.  

Eventually, the saliency is computed using the OPALS histogram etc. 

- DownsamplingPaper.py - runs the downsampling process using salinecy and voxel grid (the two are in the same file)

- downsample_compare.py - runs the comparison analysis between different types of downsampling. This file is for the downsampling paper.

Files with names of datasets run the saliency for the named dataset.

## Dependecies
- OPALS v.2.3.2 -- https://opals.geo.tuwien.ac.at/html/stable/index.html

