import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# basename = './datasets/beach/change_timeseries_tint24_nepochs123'
basename = '../../data/snowCover/change_timeseries_tint1_nepochs129_subsampled1/r25_extract_positive'
# ilist = sorted([int(os.path.basename(os.path.dirname(d)).split('_')[1])
#                 for d in glob.glob(basename+"/*")])
ilist = sorted([int(entry.split('.')[0]) for entry in  os.listdir(basename) if os.path.isfile(os.path.join(basename, entry))])

if not os.path.exists(basename + '/animation/'):
    os.mkdir(basename + '/animation/')

k = 140

for i in ilist[:k]:
    # txts = glob.glob(basename + "/change_" + str(i) + f"/{'[0-9]' * 4}.txt")
    # itrs = [int(txt.split('/')[-1][:4]) for txt in txts]
    # txt = txts[np.argmax(itrs)]
    # print(txt)
    data = np.loadtxt(basename + '/' + str(i) + '.txt')

    plt.figure(figsize=(9, 9))
    plt.title(i)
    plt.scatter(data[:, 0], data[:, 1], c=data[:, -2], cmap='coolwarm', s=1, rasterized=True)
    plt.axis('equal')
    plt.xlim([-500, 100])
    plt.ylim([-300, 350])
    plt.savefig(basename + f'/animation/{i:03d}.png', dpi=196)
    plt.close()

# for i in ilist[:k]:
#     txts = glob.glob(basename + "/change_" + str(i) + f"/{'[0-9]'*4}_extract.txt")
#     itrs = [int(txt.split('_')[-2][-4:]) for txt in txts]
#     txt = txts[np.argmax(itrs)]
#     data = np.loadtxt(txt)
#     print(txt)
#     plt.figure(figsize=(9, 9))
#     plt.title(i)
#     plt.scatter(data[:, 0], data[:, 1], c=data[:, -1], cmap='gist_heat', s=1, rasterized=True)
#     plt.axis('equal')
#     plt.xlim([-500, 100])
#     plt.ylim([-300, 350])
#     plt.savefig(basename + f'/animation/{i:03d}c.png', dpi=196)
#     plt.close()