import imageio
from glob import glob
from natsort import natsorted
from skimage.transform import resize


def create_gifv(input_glob, output_basename, fps):
    output_extensions = ["gif", "mp4"]
    input_filenames = glob(input_glob)

    input_filenames = natsorted(input_filenames)
    input_filenames = input_filenames[:1080]

    poster_writer = imageio.get_writer("{}.png".format(output_basename), mode='i')
    video_writers = [
        imageio.get_writer("{}.{}".format(output_basename, ext), mode='I', fps=fps)
        for ext in output_extensions]

    is_first = True
    count = 0
    for filename in input_filenames:
        count += 1
        if count % 5 == 0:
            print(str(count) + ' out of ' + str(len(input_filenames)))
        img = imageio.imread(filename)

        # img = resize(img, (512, 512))
        # ... processing to crop, rescale, etc. goes here

        for writer in video_writers:
            writer.append_data(img)
        if is_first:
            poster_writer.append_data(img)

        is_first = False

    for writer in video_writers + [poster_writer]:
        writer.close()


if __name__ == "__main__":

    import sys
    import os
    from scipy.misc import *
    import numpy as np

    data_dir = r'C:\SelfAttentionGAN\amc9\1141_27820\_labeled\*.png'

    create_gifv(data_dir, '.AMC_9_demo.gif', 3)

    # sat_set = []
    # for i, npy in enumerate(os.listdir(data_dir)):
    #     sat_set.append(data_dir + '/' +npy)
    #
    #
    # for i in range(len(sat_set)):
    #     X = imread(sat_set[i])
    #     X = np.expand_dims(X, axis=0)
    #     if i == 0:
    #         trX = X
    #         # trY = np.ones(X.shape[0]).astype(np.float) * i
    #     else:
    #         trX = np.concatenate([trX, X], axis=0)
    #         # trY = np.concatenate([trY, np.ones(X.shape[0]).astype(np.float) * i])
    #
    # X = trX
    # # y = trY
    #
    # fp = open("range_kalman.gif", "wb")
    # makedelta(fp, X)
    # fp.close()

    if len(sys.argv) < 3:
        print("GIFMAKER -- create GIF animations")
        print("Usage: gifmaker infile outfile")
        sys.exit(1)

    # compress(sys.argv[1], sys.argv[2])