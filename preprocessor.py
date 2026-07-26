import os
import argparse
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


def validate_feature_window_size(w_len):
    try:
        parsed = int(w_len)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature window size must be an integer") from exc
    if parsed != w_len:
        raise ValueError("feature window size must be an integer")
    if parsed < 2:
        raise ValueError("feature window size must be at least 2")
    return parsed


def feature_window_size_type(value):
    try:
        w_len = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "feature window size must be an integer"
        ) from exc
    if w_len < 2:
        raise argparse.ArgumentTypeError(
            "feature window size must be at least 2"
        )
    return w_len


def z_norm(signal, consistency_correction=1.4826, eps=1e-6):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return (signal - median) / (consistency_correction * mad + eps)

def modified_zscore(signal):
    from scipy import stats
    z = stats.zscore(signal)
    if np.isnan(z).any():
        print(f"[WARNING] zscore contains NaN, convert to zeros-like array, {z}")
        return np.zeros_like(signal)
    else:
        return z 

#def modified_zscore(signal, mad_threshold=3.5, consistency_correction=1.4826, eps=1e-6):
#    median = np.median(signal)
#    dev_from_med = np.array(signal) - median
#    mad = np.median(np.abs(dev_from_med))
#    if mad < eps:
#        # print(f"[DEBUG] MAD of signal less than 1e-6, {signal}")
#        return np.zeros_like(signal)
#    mad_score = dev_from_med / (consistency_correction * mad)
#
#    x = np.where(np.abs(mad_score) > mad_threshold)
#    x = x[0]
#
#    if len(x) > 0:
#        for i in range(len(x)):
#            if x[i] == 0:
#                mad_score[x[i]] = mad_score[x[i] + 1]
#            elif x[i] == len(mad_score) - 1:
#                mad_score[x[i]] = mad_score[x[i] - 1]
#            else:
#                mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
#    return mad_score

#def modified_zscore(signal, mad_threshold=3.5, consistency_correction=1.4826):
#    median = np.median(signal)
#    eps = 1e-6
#    dev_from_med = np.array(signal) - median
#    mad = np.median(np.abs(dev_from_med))
#    mad_score = dev_from_med / (consistency_correction * mad + eps)
#
#    x = np.where(np.abs(mad_score) > mad_threshold)
#    x = x[0]
#
#    if len(x) > 0:
#        for i in range(len(x)):
#            if x[i] == 0:
#                mad_score[x[i]] = mad_score[x[i] + 1]
#            elif x[i] == len(mad_score) - 1:
#                mad_score[x[i]] = mad_score[x[i] - 1]
#            else:
#                mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
#    return mad_score


def cut_patchs(signal, seq_length, stride, patch_size):
    split_signal = np.zeros((patch_size, seq_length), dtype="float32")
    for i in range(seq_length):
        split_signal[:, i] = signal[(i*stride):(i*stride)+patch_size]
    return split_signal


def train_normalization(data, cut, length, tile, patches, seq_length, stride, patch_size):
    step, start, segment_arr = length // tile, 0, []
    for _ in range(tile):
        for signal in data:
            if len(signal) < cut + length:
                continue

            signal = signal[cut:]
            end = start + length
            while end <= len(signal):
                segment = modified_zscore(signal[end-length:end])
                if patches:
                    segment = cut_patchs(segment, seq_length, stride, patch_size)
                segment_arr.append(segment)
                end += length
        start += step
    return np.array(segment_arr)

def train_non_normalization(data, cut, length, tile, patches, seq_length, stride, patch_size):
    step, start, segment_arr = length // tile, 0, []
    for _ in range(tile):
        for signal in data:
            if len(signal) < cut + length:
                continue

            signal = signal[cut:]
            end = start + length
            while end <= len(signal):
                segment = signal[end-length:end]
                if patches:
                    segment = cut_patchs(segment, seq_length, stride, patch_size)
                segment_arr.append(segment)
                end += length
        start += step
    return np.array(segment_arr)


def valid_normalization(data, cut, length, patches, seq_length, stride, patch_size):
    segment_arr = []
    for signal in data:
        if len(signal) < cut + length:
            continue
        segment = modified_zscore(signal[cut:cut+length])
        if patches:
            segment = cut_patchs(segment, seq_length, stride, patch_size)
        segment_arr.append(segment)
    return np.array(segment_arr)

def valid_non_normalization(data, cut, length, patches, seq_length, stride, patch_size):
    segment_arr = []
    for signal in data:
        if len(signal) < cut + length:
            continue
        segment = signal[cut:cut+length]
        if patches:
            segment = cut_patchs(segment, seq_length, stride, patch_size)
        segment_arr.append(segment)
    return np.array(segment_arr)

def comp_cum_sum(signal_arr):
    batch_size = signal_arr.shape[0]
    zero_array = np.expand_dims(np.array([0] * batch_size), axis=1)
    cum_sum_sig = np.cumsum(np.concatenate((zero_array, signal_arr), axis=1), axis=1)
    cum_sum_sig_square = np.cumsum(np.concatenate((zero_array, signal_arr ** 2), axis=1), axis=1)
    return cum_sum_sig, cum_sum_sig_square

def comp_t_stat(cum_sum_sig, cum_sum_sig_square, s_len, w_len):
    eta = np.finfo(float).eps
    t_stat = np.zeros((cum_sum_sig.shape[0], cum_sum_sig.shape[1] - 1))

    # Ensure conditions are met
    if s_len < 2 * w_len or w_len < 2:
        return t_stat

    # Compute cumulative sums for each window in a vectorized manner
    sum1 = cum_sum_sig[:, w_len:s_len - w_len + 1] - cum_sum_sig[:, :s_len - 2 * w_len + 1]
    sumsq1 = cum_sum_sig_square[:, w_len:s_len - w_len + 1] - cum_sum_sig_square[:, :s_len - 2 * w_len + 1]

    sum2 = cum_sum_sig[:, 2 * w_len:s_len + 1] - cum_sum_sig[:, w_len:s_len - w_len + 1]
    sumsq2 = cum_sum_sig_square[:, 2 * w_len:s_len + 1] - cum_sum_sig_square[:, w_len:s_len - w_len + 1]
    
    #sumsq1 = sumsq1.astype(np.float32)
    #sum1 = sum1.astype(np.float32)
    #sumsq2 = sumsq2.astype(np.float32)
    #sum2 = sum2.astype(np.float32)

    # Means for each segment
    mean1 = sum1 / w_len
    mean2 = sum2 / w_len
    
    # Calculate variances, handling minimum threshold eta
    combined_var = (sumsq1 / w_len - mean1 ** 2) + (sumsq2 / w_len - mean2 ** 2)
    combined_var = np.maximum(combined_var, eta)

    # Compute t-statistics
    delta_mean = mean2 - mean1
    t_stat_res = np.abs(delta_mean) / np.sqrt(combined_var / w_len)

    # Place the computed t-statistics into the result array
    t_stat[:, w_len:s_len - w_len + 1] = t_stat_res

    return torch.tensor(t_stat)

def diff1(sig):
    diffs = np.diff(sig, prepend=0)
    return torch.tensor(diffs)

def window_mean_std(sig, w_len):
    w_len = validate_feature_window_size(w_len)
    sig = sliding_window_view(sig, window_shape=w_len, axis=1)
    w_means = np.mean(sig, axis=2)
    w_stds = np.std(sig, axis=2)

    pad_total = w_len - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    w_means = np.pad(w_means, ((0, 0), (pad_left, pad_right)))
    w_stds = np.pad(w_stds, ((0, 0), (pad_left, pad_right)))

    return torch.tensor(w_means), torch.tensor(w_stds)

#def add_features(signals, norm=False, s_len=3000, w_len=3):
#
#    signal_arr = torch.tensor(signals, dtype=torch.float32)
#    
#    cum_sum_sig, cum_sum_sig_square = comp_cum_sum(signal_arr)
#    t_stat = comp_t_stat(cum_sum_sig, cum_sum_sig_square, s_len, w_len)
#    diff = diff1(signal_arr)
#    w_means, w_stds = window_mean_std(signal_arr, w_len=w_len)
#    signals_res = torch.stack([signal_arr, diff, w_means, w_stds, t_stat], dim=1)
        # (5, 3000)

#    return signals_res
def add_features(signals, norm=True, s_len=3000, w_len=3):
    """
    channel 0: raw current
    channel 1: diff
    channel 2: window mean
    channel 3: window std
    channel 4: t-stat
    channel 5: z-score current

    signals: (N, L) numpy array
    return:  (N, 6, L) torch tensor
    """
    w_len = validate_feature_window_size(w_len)
    signal_arr = torch.as_tensor(signals, dtype=torch.float32)  # (N, L)
    N, L = signal_arr.shape

    # channel 0
    raw = signal_arr

    # channel 1
    diff = diff1(raw)

    # channel 2 / 3
    w_means, w_stds = window_mean_std(raw, w_len=w_len)

    # channel 4
    cum_sum_sig, cum_sum_sig_square = comp_cum_sum(raw)
    t_stat = comp_t_stat(cum_sum_sig, cum_sum_sig_square, s_len, w_len)

    # channel 5
    z_signal = torch.zeros_like(raw)
    if norm:
        for i in range(N):
            z = modified_zscore(raw[i].cpu().numpy())
            z_signal[i] = torch.from_numpy(z)

    # stack in strict channel order
    signals_res = torch.stack(
        [raw, diff, w_means, w_stds, t_stat, z_signal],
        dim=1
    )  # (N, 6, L)

    return signals_res



def numpy_to_tensor_in_batches(np_array, batch_size=10000, dtype=torch.float32, device='cpu'):
    n_rows = np_array.shape[0]
    tensor_list = []
    print("[DEBUG]Input shape: ", np_array.shape)
    for i in range(0, n_rows, batch_size):
        j = min(i + batch_size, n_rows)
        batch = torch.from_numpy(np_array[i:j].astype(np.float32)).to(dtype=dtype, device=device)
        tensor_list.append(batch)

    full_tensor = torch.cat(tensor_list, dim=0)
    print("[DEBUG]Output shape: ", full_tensor.shape)
    return full_tensor

if __name__ == '__main__':
    # Get command arguments
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument("--data_folder", '-d', type=str, required=True, help="Path to the dataset folder that contains train, valid, test files (.npy)")
    parser.add_argument("--cut", '-c', type=int, default=1500, help="Electrical signal length to be cut, default 1500")
    parser.add_argument("--tiling_fold", '-tf', type=int, default=3, help="Number of tiles, default 3")
    parser.add_argument("--length", '-l', type=int, default=3000, help="The length of the sliding window, default 3000")
    parser.add_argument("--patches", '-patches', action='store_true', help="Convert electrical signals into patches, default False")
    parser.add_argument("--seq_length", '-sl', type=int, default=299, help="Sequence length after patch, default 299")
    parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
    parser.add_argument("--patch_size", '-ps', type=int, default=16, help="The size of patch, default 16")
    args = parser.parse_args()

    # Print parameter information
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Load dataset
    print("\nLoad dataset!")
    train_data = np.load(os.path.join(args.data_folder, "train.npy"), allow_pickle=True)
    print(f"Successfully loaded training set from {os.path.join(args.data_folder, 'train.npy')}, shape: {train_data.shape}")
    valid_data = np.load(os.path.join(args.data_folder, "valid.npy"), allow_pickle=True)
    print(f"Successfully loaded validation set from {os.path.join(args.data_folder, 'valid.npy')}, shape: {valid_data.shape}")

    # Normalize using modified Z-score and cut signal
    print("\nPreprocess dataset!")
    train_data = train_normalization(train_data, args.cut, args.length, args.tiling_fold,
                                  args.patches, args.seq_length, args.stride, args.patch_size)
    print(f"Successfully preprocessed training set, shape: {train_data.shape}")
    valid_data = valid_normalization(valid_data, args.cut, args.length,
                                  args.patches, args.seq_length, args.stride, args.patch_size)
    print(f"Successfully preprocessed validation set, shape: {valid_data.shape}")

    # Create training data and validation data (npy)
    print("\nSave data!")
    np.save(os.path.join(args.data_folder, "train_preprocessed.npy"), train_data)
    print(f"Successfully saved preprocessed training set to {os.path.join(args.data_folder, 'train_preprocessed.npy')}")
    np.save(os.path.join(args.data_folder, "valid_preprocessed.npy"), valid_data)
    print(f"Successfully saved preprocessed validation set to {os.path.join(args.data_folder, 'valid_preprocessed.npy')}")
