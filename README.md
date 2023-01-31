# MorE3D 

## Intorduction -- in short

This project aims to develop tools for the processing of 3D point clouds for geomorphological applications. 

## Introduction -- detailed

Landscape reshaping processes exert great impact on wildlife, humans and their habitats. Over the last decades, the use of high-resolution laser scanning technologies to document, monitor, and analyse the morphological signatures these processes imprint on the terrain has improved our understanding of their nature and helped develop strategies to avert hazards and deliver a more sustainable future. However, the fragmented nature of current research in this area prevents researchers from taking full advantage of the large amounts of available data. The EU-funded MorE3D project will introduce a processing framework that will combine the data with geoscientist practices to improve the interaction with the data by highlighting features and offering analyses necessary for the assessment of natural processes.

## This repository 

### Saliency 


- DM_saliency.py - use of pyDM to compute the saliency. Two processors are defined: 
    
    - curvature computation
    - dk and dn computation 
- BallTreePointSet.py - a class for the ball tree data structure.  

Eventually, the saliency is computed using the OPALS histogram etc. 

#### Citing

When using this code, please cite: 

@Article{Arav.Filin2022,
  author    = {Reuma Arav and Sagi Filin},
  journal   = {{ISPRS} Journal of Photogrammetry and Remote Sensing},
  title     = {A visual saliency-driven extraction framework of smoothly embedded entities in 3D point clouds of open terrain},
  year      = {2022},
  month     = {jun},
  pages     = {125--140},
  volume    = {188},
  doi       = {10.1016/j.isprsjprs.2022.04.003},
  publisher = {Elsevier {BV}},
}


### Downsampling

- DownsamplingPaper.py - runs the downsampling process using salinecy and voxel grid (the two are in the same file)

- downsample_compare.py - runs the comparison analysis between different types of downsampling. 

Files with names of datasets run the saliency for the named dataset.

#### Citing

When using this code, please cite:


@ARTICLE{Arav.etal2022,
  author={Arav, Reuma and Filin, Sagi and Pfeifer, Norbert},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Content-Aware Point Cloud Simplification of Natural Scenes}, 
  year={2022},
  volume={60},
  number={},
  pages={1-12},
  doi={10.1109/TGRS.2022.3208348}
  }

### Level set 

- run_levelsets.py - runs 3D level set extraction in an input point cloud based on a chosen feature. The level set will delineate and extract the 3D shapes that their features are similar inside and outside the curve. 

- levelset_func.py - the level set function that runs in the background of "run_levelsets.py". 

- pcls_timeseries.ipnyb - a jupyter notebook that creates a video of the extraction if it is based on a time series (extraction through time). 

#### Citing

When using this code, please cite:

@article{Arav.etal2022b,
author={Arav,R. and Pöppl,F. and Pfeifer,N.},
year={2022},
title={A POINT-BASED LEVEL-SET APPROACH FOR THE EXTRACTION OF 3D ENTITIES FROM POINT CLOUDS – APPLICATION IN GEOMORPHOLOGICAL CONTEXT},
journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
volume={-2-2022},
pages={95-102},
isbn={21949042},
doi={10.5194/isprs-annals-V-2-2022-95-2022}
} 

### Examples
- run_saliency.py - main running code. For now it runs the entire saliency computation process using OPALS. 



### Dependecies
- OPALS v.2.3.2 -- https://opals.geo.tuwien.ac.at/html/stable/index.html
- Scikit-image v.0.19.3 

### Note

This repository is part of MorE3D — an ongoing project carried at TU Wien and funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant.
