"""
Create a new las file with the rates as an attribute
"""

import laspy
import os
import levelsets_func as ls
import numpy as np


def compute_rate(data, args):

    field1 = data[args[0]]
    field2 = data[args[1]]

    print(f"processing '{field1}'/'{field2}'")
    return field2-field1


def create_las(points, fields):
    """
    Creates new las file with fields

    :type fields: dict
    :type las_points: np.ndarray
    """
    new_hdr = laspy.LasHeader(version="1.4", point_format=6)

    extrabytesparam = [laspy.ExtraBytesParams(name=name, type='float64') for name in fields.keys()]
    new_hdr.add_extra_dims(extrabytesparam)
        # new_hdr.add_extra_dims(laspy.ExtraBytesParams(name=name, type='float64'))
    # new_las = laspy.create(point_format=6)
    # new_las.header = new_hdr
    new_las = laspy.LasData(new_hdr)
    new_las.x = points[:,0]
    new_las.y = points[:,1]
    new_las.z = points[:,2]
    for name in fields.keys():
        new_las[name] = fields[str(name)]
    # new_las.__dict__.update(fields)


    return new_las

# ------------- run -----------------

in_file = '../data/snowCover/change_timeseries_tint1_nepochs129_subsampled1.las'
fields = [f'change_{i}' for i in range(0, 126)]
# fields = ['zeros','change_125']
t = 25  # use epoch `n` and `n+t`

# maximal number of point holding "nan" values
max_nan = 60000
# new filename to save the new las
new_file = in_file.split('.las')[0]  + '_rate' + str(t) + '.las'

print(f"reading file '{in_file}'")
data = ls.read_las(in_file, fields)
rates = dict()
i=0
for epoch in zip(fields[:-t], fields[t:]):

    rate = compute_rate(data, epoch)
    rate /= t  # change rate
    print(np.sum(np.isnan(rate)))

    if np.sum(np.isnan(rate)) > max_nan:
        continue
    field_name = 'r' + epoch[0].split('_')[1] #+ '_' + epoch[1].split('_')[1]
    new_field = {field_name: rate}
    rates.update(new_field)

las = create_las(data["xyz"], rates)
las.write(new_file)