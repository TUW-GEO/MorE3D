from opals import Import, Grid, Shade, Normals, Export
from opals import View
from opals import pyDM, AddInfo
import opals
import cProfile, pstats, io
from os.path import exists

import DM_saliency
from DM_saliency import DM_nonparam_curvature
# from DM_spatial_query_point_index import DM_spatial_query_point_index
# from DM_attributes_statistic import DM_attributes_statistic

if __name__ == '__main__':
    # input
    # pr = cProfile.Profile()
    # pr.enable()
    folder ='../data/cave1/'
    file = 'cave1_1.3M'
    lasFile_orig = folder + file + '.las'
    newODM = False  # whether to compute surface features again, or load from disk

    # define output ascii format
    exp_ascii = Export.Export()
    exp_ascii.oFormat = 'normalsEchoRatioAscii.xml'


    # define output las format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_curvature_saliency.xml'


    # preliminary parameters
    normals_rad = .1

    curvature_rad = normals_rad
    curvature_roughness = 0.0001
    odmFile = folder + file + '_n' + str(normals_rad) + '.odm'

    # saliency parameters
    s_rho = .3
    s_sigma = s_rho / 3
    s_rad = s_rho * 3
    noise_curvature = 0.0001
    noise_normal = 0.001

    if not exists(odmFile):
        newODM = True

    if newODM:

        imp = Import.Import()
        imp.inFile = lasFile_orig
        imp.outFile = odmFile
        imp.commons.screenLogLevel = opals.Types.LogLevel.warning
        imp.run()

        # pyDM.Datamanager.load parameter: filename(string), readOnly(bool) threadSafety(bool)
        # grd = Grid.Grid(inFile=imp.outFile, outFile='Kaunertal_420K.tif', gridSize = 0.2,
        #                 interpolation=opals.Types.GridInterpolator.movingPlanes)
        # grd.run()
        #
        # Shade.Shade(inFile=grd.outFile).run()
        # #

        Normals.Normals(inFile=odmFile,
                        searchMode=opals.Types.SearchMode.d3, direction=opals.Types.NormalsDirection.toOrigin, neighbours=10000,
                        searchRadius=normals_rad, storeMetaInfo=opals.Types.NormalsMetaInfo.medium).run()

        DM_nonparam_curvature(odmFile, 'sphere', roughness=curvature_roughness, radius=curvature_rad)
#        opals.AddInfo.AddInfo(inFile=odmFile, attribute=f'_dip=asin((NormalX*NormalX + NormalY*NormalY)/(NormalX*NormalX + NormalY*NormalY + NormalZ * NormalZ + 0.000000001))').run()

        # exp.outFile = folder + 'curvature/' + file + 'n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rough' + str(curvature_roughness) +'.las'
        # exp.inFile = odmFile
        # exp.run()
        # View.View(inFile=odmFile).run()
    DM_saliency.DM_dk_dn(odmFile, 'sphere', rho_neighbourhood=s_rho, sigma_neighbourhood=s_sigma, radius=s_rad,
                         sigma_curvature=noise_curvature, sigma_normal=noise_normal)
    # exp.outFile = folder + 'saliency/' + file + 'n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rho' + str(
    #     s_rho) + '_kn_noise' + str(noise_normal) + 'dndk.las'
    # exp.inFile = odmFile
    # exp.run()

    DM_saliency.DM_saliency(odmFile, curvature_weight=0.3)

    exp.outFile = folder + 'saliency/' + file + 'n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rho' + str(s_rho) + '_kn_noise' + str(noise_normal) + '.las'
    exp.inFile = odmFile
    exp.run()

    # View.View(inFile=odmFile).run()
    # pr.disable()
    # s = io.StringIO()
    # stats_out = 'profile.stats'
    # pr.dump_stats(stats_out)
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

