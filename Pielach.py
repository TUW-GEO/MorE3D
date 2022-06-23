from opals import Import, Grid, Shade, Normals, Export
from opals import View
from opals import pyDM
import opals
from os.path import exists

import DM_saliency
from DM_saliency import DM_nonparam_curvature

def remove_nonplanar(inFile, eigen3_thresh):
    """
    Classifies the point cloud from the odm to planar and non planar.

    :TODO: change so it won't build a new las but will classify within the odm to use with filter. ASK JOHANNES HOW!

    :param inFile: path to the odm to be classified
    :param eigen3_thresh: threshold for eigenvalue3 to be zero

    :return: a new odm without the non planar points
    """
    try:
        dm = pyDM.Datamanager.load(inFile, False, False)
    except IOError as e:
        print(e)
        return

        # initialize an empty layout
    # lf = pyDM.AddInfoLayoutFactory()
    # lf.addColumn(pyDM.ColumnSemantic.NormalEigenvalue1)  # column 0
    # lf.addColumn(pyDM.ColumnSemantic.NormalEigenvalue2)  # column 1
    # lf.addColumn(pyDM.ColumnSemantic.NormalEigenvalue3)  # column 2
    #
    # layout = lf.getLayout()

    # opals.AddInfo.AddInfo(inFile=inFile, attribute=f'_normed_dk(float) = (_dk - {min_k}) / {diff_k} + {min_k}').run() # normalize to values (0,1)
    filter = pyDM.Filter(f'NormalEigenvalue1 * NormalEigenvalue2 > 0 & abs(NormalEigenvalue3) < {eigen3_thresh}')


if __name__ == '__main__':
    # input
    folder = '../data/Pielach/'
    file = 'Pielach_55.7K'
    lasFile_orig = folder + file + '.las'

    newODM = True # whether to compute surface features again, or load from disk
    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_curvature_saliency.xml'

    # preliminary parameters
    normals_rad = .3

    curvature_rad = normals_rad
    curvature_roughness = 0.015

    # saliency parameters
    s_rho = .5
    s_sigma = s_rho / 3
    s_rad = s_rho * 2
    noise_curvature = 0.07
    noise_normal = 0.02

    odmFile = folder + file + '_n' + str(normals_rad) + '.odm'

    eigen3_thresh = 0.001

    if not exists(odmFile):
        newODM = True

    if newODM:
        imp = Import.Import()
        imp.inFile = lasFile_orig
        imp.outFile = odmFile
        imp.commons.screenLogLevel = opals.Types.LogLevel.warning
        imp.run()

        Normals.Normals(inFile=odmFile,
                        searchMode=opals.Types.SearchMode.d3, direction=opals.Types.NormalsDirection.toOrigin,
                        neighbours=10000,
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

    DM_saliency.DM_saliency(odmFile, curvature_weight=0.5)

    exp.outFile = folder + 'saliency/' + file + 'n' + str(normals_rad) + '_k' + str(curvature_rad) + '_rho' + str(
        s_rho) + '_kn_noise' + str(noise_normal) + '.las'
    exp.inFile = odmFile
    exp.run()

