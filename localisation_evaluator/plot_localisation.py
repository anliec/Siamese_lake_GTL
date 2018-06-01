import numpy as np
import pickle
import os
import argparse
import matplotlib
# import cProfile
# import pstats
# import io
# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MAX_OFFSET = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir',
                        default="out",
                        type=str,
                        dest="out_path")
    parser.add_argument('-r', '--result-path',
                        required=True,
                        type=str,
                        dest="result_path")
    args = parser.parse_args()

    with open(args.result_path, 'rb') as handle:
        results = pickle.load(handle)

    results = sorted(results, key=lambda x: str.lower(x[0] + x[1]))

    old_d1, old_s1, old_d2, old_s2 = None, None, None, None
    current_file = None
    results_array = None

    os.makedirs(args.out_path, exist_ok=True)

    # pr = cProfile.Profile()
    c = mcolors.ColorConverter().to_rgb
    color_map = make_colormap([(0.0, 1.0, 0.0), c('white'), 0.5, c('white'), (1.0, 0.0, 0.0)])

    def float_str_to_int(txt: str):
        return int(float(txt))

    for i, (file1, file2, score) in enumerate(results):
        # pr.enable()
        print(i, '/', len(results), '(', i * 100 // len(results), '%)', end='\r')
        d1, s1, _, _, offset1, d2, s2, _, _, offset2 = \
            list(map(float_str_to_int, os.path.splitext(os.path.split(file1)[1])[0].split('_')))
        if d1 != old_d1 or s1 != old_s1 or d2 != old_d2 or s2 != old_s2:
            # if we have already read some results, plot them to a file
            if current_file is not None:
                plt.imshow(results_array, extent=(-MAX_OFFSET, MAX_OFFSET, -MAX_OFFSET, MAX_OFFSET), cmap=color_map)
                plt.savefig(os.path.join(args.out_path, current_file))
            # reset results array as well as file name
            current_file = "_".join(map(str, [d1, s1, d2, s2])) + ".png"
            results_array = np.zeros(shape=(2 * MAX_OFFSET + 1, 2 * MAX_OFFSET + 1), dtype=float)
        # add current result to array
        results_array[offset1 + MAX_OFFSET, offset2 + MAX_OFFSET] = score
        old_d1, old_s1, old_d2, old_s2 = d1, s1, d2, s2
        # pr.disable()
        if i > 1000:
            break

    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())


# def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
#     """
#     low and high are colors that will be used for the two
#     ends of the spectrum. they can be either color strings
#     or rgb color tuples
#     """
#     c = mcolors.ColorConverter().to_rgb
#     if isinstance(low, str):
#         low = c(low)
#     if isinstance(high, str):
#         high = c(high)
#     return make_colormap([low, c('white'), 0.5, c('white'), high])


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


if __name__ == "__main__":
    main()

