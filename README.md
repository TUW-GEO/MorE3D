# MorE3D 

## Intorduction -- in short

This project aims to develop tools for the processing of 3D point clouds for geomorphological applications. 

## Introduction -- detailed

Landscape reshaping processes exert great impact on wildlife, humans and their habitats. Over the last decades, the use of high-resolution laser scanning technologies to document, monitor, and analyse the morphological signatures these processes imprint on the terrain has improved our understanding of their nature and helped develop strategies to avert hazards and deliver a more sustainable future. However, the fragmented nature of current research in this area prevents researchers from taking full advantage of the large amounts of available data. The EU-funded MorE3D project will introduce a processing framework that will combine the data with geoscientist practices to improve the interaction with the data by highlighting features and offering analyses necessary for the assessment of natural processes.

## This repository 

### Saliency 

- main.py - main running code. For now it runs the entire saliency computation process using OPALS. 

- DM_saliency.py - use of pyDM to compute the saliency. Two processors are defined: 
    
    - curvature computation
    - dk and dn computation 
- BallTreePointSet.py - a class for the ball tree data structure.  

Eventually, the saliency is computed using the OPALS histogram etc. 

### Subsampling

- DownsamplingPaper.py - runs the downsampling process using salinecy and voxel grid (the two are in the same file)

- downsample_compare.py - runs the comparison analysis between different types of downsampling. This file is for the downsampling paper.

Files with names of datasets run the saliency for the named dataset.

### Dependecies
- OPALS v.2.3.2 -- https://opals.geo.tuwien.ac.at/html/stable/index.html

### Note

This repository is part of MorE3D — an ongoing project carried at TU Wien and funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant.
