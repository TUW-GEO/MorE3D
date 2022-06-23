from opals import Import, Grid, Shade, Normals, Export
from opals import View
from opals import pyDM
import opals

import DM_saliency
from DM_saliency import DM_nonparam_curvature

if __name__ == '__main__':
    # input
    folder = '../data/leopards/'
    file = 'leopards225K'
    lasFile_orig = folder + file + '.las'
    odmFile = folder + file + '.odm'

    newODM = True # whether to compute surface features again, or load from disk

    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_curvature_saliency.xml'

    # preliminary parameters
    normals_rad = .03

    curvature_rad = .02
    curvature_roughness = 0.005

    # saliency parameters
    s_rho = 0.05
    s_sigma = s_rho / 3
    s_rad = s_rho * 2
    noise_curvature = 0.001
    noise_normal = 0.01

    if newODM:
        imp = Import.Import()
        imp.inFile = lasFile_orig
        imp.outFile = odmFile
        imp.commons.screenLogLevel = opals.Types.LogLevel.warning
        imp.run()

    # pyDM.Datamanager.load parameter: filename(string), readOnly(bool) threadSafety(bool)
    # grd = Grid.Grid(odmFile, outFile=file + '.tif', gridSize=0.01,
    #                 interpolation=opals.Types.GridInterpolator.movingPlanes)
    # grd.run()
    # #
    # Shade.Shade(inFile=grd.outFile).run()


        Normals.Normals(inFile=odmFile,
                    searchMode=opals.Types.SearchMode.d3, direction=opals.Types.NormalsDirection.upwards,
                    searchRadius=normals_rad, storeMetaInfo=opals.Types.NormalsMetaInfo.medium).run()

    # DM_attributes_statistic(dm)
        DM_nonparam_curvature(odmFile, 'sphere', roughness=curvature_roughness, radius=curvature_rad)

    # Compute the saliency
    DM_saliency.DM_dk_dn(odmFile, 'sphere', rho_neighbourhood=s_rho, sigma_neighbourhood=s_sigma, radius=s_rad,
                              sigma_curvature=noise_curvature, sigma_normal=noise_normal)
    DM_saliency.DM_saliency(odmFile, curvature_weight=0.3)

    exp.outFile = folder + 'saliency/' + file + '_n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rho' + \
                  str(s_rho) + '_kn_noise' + str(noise_normal) + '.las'
    exp.inFile = odmFile
    exp.run()

    View.View(inFile=odmFile).run()

