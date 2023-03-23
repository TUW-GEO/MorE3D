"""
===================================================================
                            MorE3d
===================================================================
Example for running saliency using OPALS
"""
import opals
from opals import Import, Grid, Shade, Normals, Export
from opals import View
from opals import pyDM
import opals

import DM_saliency
from DM_saliency import DM_nonparam_curvature

if __name__ == '__main__':
    # input
    file = '/home/rarav/PycharmProjects/data/Gesher/gesher_wall4_RGB.las'

    lasFile_orig = file + '.las'
    odmFile = file + '.odm'

    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_curvature_saliency.xml'

    # preliminary parameters
    normals_rad = .1

    curvature_rad = .1
    curvature_roughness = 0.0001

    # saliency parameters
    s_rho = .025
    s_sigma = s_rho / 3
    s_rad = s_rho * 2
    noise_curvature = 0.0001
    noise_normal = 0.0001

    # Import las file
    imp = Import.Import()
    imp.inFile = lasFile_orig
    imp.outFile = odmFile
    imp.commons.screenLogLevel = opals.Types.LogLevel.warning
    imp.run()

    Normals.Normals(inFile=odmFile,
                    searchMode=opals.Types.SearchMode.d3, direction=opals.Types.NormalsDirection.upwards,
                    searchRadius=normals_rad, neighbours=300, storeMetaInfo=opals.Types.NormalsMetaInfo.medium).run()
    DM_nonparam_curvature(odmFile, 'sphere', roughness=0.0001, radius=curvature_rad)
    DM_saliency.DM_dk_dn(odmFile, 'sphere', rho_neighbourhood=s_rho, sigma_neighbourhood=s_sigma, radius=s_rad,
                              sigma_curvature=noise_curvature, sigma_normal=noise_normal)
    DM_saliency.DM_saliency(odmFile, curvature_weight=0.3)

    exp.outFile = file + 'n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rho' + str(s_rho) + '_kn_noise' + str(noise_normal) + '.las'
    exp.inFile = odmFile
    exp.run()

    View.View(inFile=odmFile).run()

