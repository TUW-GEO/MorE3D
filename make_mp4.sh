#/bin/bash

fps=5

ffmpeg -framerate $fps -i ${IMAGE_FOLDER}/%03d.png -c:v libx264 -r $fps -pix_fmt yuv420p ${IMAGE_FOLDER}/timeseries_delineation.mp4
ffmpeg -framerate $fps -i ${IMAGE_FOLDER}/%03dc.png -c:v libx264 -r $fps -pix_fmt yuv420p ${IMAGE_FOLDER}/timeseries_changes.mp4

