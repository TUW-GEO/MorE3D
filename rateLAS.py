"""
Create a new las file with the rates as an attribute
"""

import laspy
import os
import levelsets_func as ls


def compute_rate(args):
    field1, field2 = args

    print(f"processing '{field1}'/'{field2}'")
    return field2-field1

def create_las(filename, las_points):
    """
    Creates an empty las file
    """
    new_hdr = laspy.LasHeader(version="1.4", point_format=6)
    new_las = laspy.LasData(new_hdr)
    new_las.points = las_points
    new_las.write(filename)

def update_las(las, field, field_name):
    """
    Adds fields to las
    """
    las.add_extra_dim(laspy.ExtraBytesParams(name = field_name, type = "f8"))
    las[field_name] = field
    return las


in_file = '../data/snowCover/change_timeseries_tint1_nepochs129_subsampled1nonan.las'
fields = [f'change_{i}' for i in range(0, 126)]
# fields = ['zeros','change_125']
t = 1  # use epoch `n` and `n+t`

# dir to save the new las
base_dir = os.path.join(os.path.dirname(in_file),
                        os.path.splitext(os.path.basename(in_file))[0] )

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

new_file = base_dir + os.path.splitext(os.path.basename(in_file))[0]  + 'rates.las'

las = laspy.read(in_file)
create_las(new_file, las.points)
new_las = las.open(new_file)

for _ in zip(fields[:-t], fields[t:]):
    rate = compute_rate(_)
    new_las = update_las(new_las, rate, name_rate)

