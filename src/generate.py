import argparse
import GuitarNeuralNet
import torch
import pickle
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def generate(args):
    device = torch.device("mps")
    model = GuitarNeuralNet.GuitarNeuralNet.load_from_checkpoint("./model/model.ckpt")
    model.to(device)
    model.eval()

    training_data = pickle.load(open(args.train_data, "rb"))
    mean, std = training_data["mean"], training_data["std"]

    in_rate, data = wavfile.read(args.input)
    assert in_rate == 44100, "input data needs to be 44.1 kHz"
    sample_size = int(in_rate * args.sample_time)
    length = len(data) - len(data) % sample_size

    # split into samples
    input = data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    # standardize input
    input = (input - mean) / std

    # pad each sample with the previous one for temporal context
    prev_samples = np.concatenate((np.zeros_like(input[0:1]), input[:-1]), axis=0)
    new_input = np.concatenate((prev_samples, input), axis=2)

    predicted = []
    batches = new_input.shape[0] // args.batch_size
    for x in tqdm(np.array_split(new_input, batches)):
        input_tensor = torch.from_numpy(x).to(device).float()
        predicted.append(model(input_tensor).cpu().numpy())

    predicted_np = np.concatenate(predicted)
    predicted_np = predicted_np[:, :, -input.shape[2] :]

    print(len(predicted_np))
    print(np.max(np.abs(len(predicted_np))))
    print(int(1 // args.sample_time))

    wavfile.write(args.output, 44100, predicted_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pedalnet.ckpt")
    parser.add_argument("--train_data", default="data.pickle")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    print(args)
    generate(args)
