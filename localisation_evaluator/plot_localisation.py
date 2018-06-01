import numpy as np
import pickle
import os
import argparse
import matplotlib
# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    f1, f2 = None, None
    current_file = None
    results_array = None

    os.makedirs(args.out_path, exist_ok=True)

    def float_str_to_int(txt: str):
        return int(float(txt))

    for i, (file1, file2, score) in enumerate(results):
        print(i, '/', len(results), '(', i * 100 // len(results), '%)')
        d1, s1, _, _, offset1, d2, s2, _, _, offset2 = \
            list(map(float_str_to_int, os.path.splitext(os.path.split(file1)[1])[0].split('_')))
        if f1 != file1 or f2 != file2 or f1 is None or f2 is None:
            # if we have already read some results, plot them to a file
            if current_file is not None:
                plt.imshow(results_array, extent=(-MAX_OFFSET, MAX_OFFSET, -MAX_OFFSET, MAX_OFFSET))
                plt.savefig(os.path.join(args.out_path, current_file))
            # reset results array as well as file name
            current_file = "_".join(map(str, [d1, s1, d2, s2])) + ".png"
            results_array = np.zeros(shape=(2 * MAX_OFFSET + 1, 2 * MAX_OFFSET + 1), dtype=float)
        # add current result to array
        results_array[offset1 + MAX_OFFSET, offset2 + MAX_OFFSET] = score
        f1 = file1
        f2 = file2


if __name__ == "__main__":
    main()

