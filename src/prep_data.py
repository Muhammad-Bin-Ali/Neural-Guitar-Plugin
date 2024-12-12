import argparse
from scipy.io import wavfile
import numpy as np
import pickle


def prep_data(args):
    in_rate, in_data = wavfile.read(args.input_file)
    out_rate, out_data = wavfile.read(args.output_file)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
    y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])  # splitting audio into three sections

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)

    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["x_train"].mean()) / d["x_train"].std()  # mean-variance normalization

    pickle.dump(d, open(args.data, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)

    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--sample_time", type=float, default=100e-3)

    args = parser.parse_args()
    prep_data(args)
