import torch


def mirror_pitch_axis(x):
    """
    Mirror the y-axis and stack it after the original.

    Example:
        C D E F G A B
        ->
        C D E F G A B C B A G F E D C

    Args:
        x: input tensor
        y_dim: dimension corresponding to the pitch/y-axis

    Returns:
        Tensor with the y-axis extended in both directions.
    """
    assert x.ndim == 2
    assert x.shape[0] == 12
    # Reverse entire pitch axis: Si La ... Re Do
    x_reverse = torch.flip(x[:-1, :], dims=[0])

    return torch.cat([x, x_reverse], dim=0)


def shift_pitch_hpcp(x, p=1.0, max_pitch_shift=6):
    if torch.rand(1).item() < p:
        shift = torch.randint(-max_pitch_shift, max_pitch_shift + 1, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=0)
        return x
    else:
        return x
    

def shift_mask_hpcp_time(x, p=1.0, min_shift=0, max_shift=300, mask_width=(0, 300)):
    if torch.rand(1).item() < p:
        x = x.clone()
        T = x.shape[-1]

        shift = torch.randint(min_shift, max_shift + 1, (1,)).item()
        if torch.rand(1).item() < 0.5:
            shift = -shift

        out = torch.zeros_like(x)

        if shift > 0: # assign shifted part to zero tensor
            out[..., shift:] = x[..., :T-shift]
        else:
            s = -shift
            out[..., :T-s] = x[..., s:]
        x = out
        return x
    else:
        return x
    

# def partial_hpcp(x, p=1.0, min_ratio=0.25, max_ratio=0.5):
#     if torch.rand(1).item() < p:
#         T = x.shape[-1]
#         keep_len = int(torch.empty(1).uniform_(min_ratio, max_ratio).item() * T) # keep min~max ratio
#         start = torch.randint(0, T - keep_len + 1, (1,)).item() # unmask start
#         out = torch.zeros_like(x)
#         out[..., start:start + keep_len] = x[..., start:start + keep_len]

#         return out
#     else:
#         return x
    

def partial_gap_hpcp(x, p=1.0, min_ratio=0.25, max_ratio=0.5, min_chunk=50, max_chunk=200, min_gap=20, max_gap=150):
    if torch.rand(1).item() < p:
        T = x.shape[-1]

        keep_len = int(torch.empty(1).uniform_(min_ratio, max_ratio).item() * T)

        out = torch.zeros_like(x)

        remaining = keep_len
        pos = torch.randint(0, max(1, min_gap + 1), (1,)).item()

        while remaining > 0 and pos < T:
            chunk_len = torch.randint(min_chunk, max_chunk + 1, (1,)).item()
            chunk_len = min(chunk_len, remaining, T - pos)

            if chunk_len <= 0:
                break

            out[..., pos:pos + chunk_len] = x[..., pos:pos + chunk_len]

            remaining -= chunk_len

            gap = torch.randint(min_gap, max_gap + 1, (1,)).item()
            pos += chunk_len + gap

        return out

    return x
    

def cut_hpcp(x, p=1.0, ratio=0.25):
    if torch.rand(1).item() < p:
        T = x.shape[-1]
        # keep_len = int(torch.empty(1).uniform_(ratio).item() * T) # keep min~max ratio
        start = torch.randint(0, T - ratio + 1, (1,)).item() # unmask start
        out = x[..., start:start + ratio]

        return out
    else:
        return x
    
    
# def keep_partial_hpcp(x, min_ratio=0.5, max_ratio=0.8):
#     """
#     Keep one contiguous HPCP segment and zero the rest.
#     x: [hpcp_bins, T]
#     """
#     T = x.shape[-1]

#     keep_ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
#     keep_len = max(1, int(keep_ratio * T))

#     start = torch.randint(0, T - keep_len + 1, (1,)).item()

#     out = torch.zeros_like(x)
#     out[..., start:start+keep_len] = x[..., start:start+keep_len]

#     return out
    

# Old HPCP Augmentation
# def augment_hpcp(x, max_pitch_shift=6, max_time_shift=2000, p=1.0, max_mask_width=1000):
#     if torch.rand(1).item() < p: # pitch class shift: (-2, -1, 0, 1, 2) -< (-)
#         shift = torch.randint(-max_pitch_shift, max_pitch_shift + 1, (1,)).item()
#         x = torch.roll(x, shifts=shift, dims=0)

#     if torch.rand(1).item() < p: # time masking: mask contiguous chunk
#         T = x.shape[-1]
#         width = torch.randint(1, min(max_mask_width, T) + 1, (1,)).item()
#         start = torch.randint(0, T - width + 1, (1,)).item()

#         x = x.clone()
#         x[..., start:start + width] = 0

#     if torch.rand(1).item() < p: # time shifting
#         shift = torch.randint(-max_time_shift, max_time_shift + 1, (1,)).item()
#         x = torch.roll(x, shifts=shift, dims=-1)

#     return x


# def mix_same_song_segments_hpcp(x, p=0.2, seg_len=800): # averaging two HPCP segments
#     if torch.rand(1).item() < p:
#         T = x.shape[-1]

#         if T < 2 * seg_len:
#             return x

#         s1 = torch.randint(0, T - seg_len + 1, (1,)).item()
#         s2 = torch.randint(0, T - seg_len + 1, (1,)).item()

#         seg1 = x[..., s1:s1 + seg_len]
#         seg2 = x[..., s2:s2 + seg_len]

#         return 0.5 * seg1 + 0.5 * seg2
#     else:
#         return x


# def splice_same_song_hpcp(x, p=0.2, seg_len=500): # concat or replacing temportal regions of the same song
#     if torch.rand(1).item() < p:
#         T = x.shape[-1]

#         if T < 2 * seg_len:
#             return x

#         s1 = torch.randint(0, T - seg_len + 1, (1,)).item()
#         s2 = torch.randint(0, T - seg_len + 1, (1,)).item()

#         x_aug = x.clone()
#         x_aug[..., s1:s1 + seg_len] = x[..., s2:s2 + seg_len]

#         return x_aug
#     else:
#         return x
    

# CQT Augmentation
# def shift_pitch_cqt(x, bins_per_octave=12, p=1.0, max_semitones=6):
#     if torch.rand(1).item() < p:
#         semitone_shift = torch.randint(-max_semitones, max_semitones + 1, (1,)).item()
#         bin_shift = semitone_shift * (bins_per_octave // 12)
#         out = torch.zeros_like(x)

#         if bin_shift > 0:
#             out[..., bin_shift:, :] = x[..., :-bin_shift, :]
#         elif bin_shift < 0:
#             s = -bin_shift
#             out[..., :-s, :] = x[..., s:, :]
#         else:
#             out = x

#         return out
#     else:
#         return x
    

# def shift_mask_cqt_time(x, p=1.0, min_shift=0, max_shift=300):
#     if torch.rand(1).item() < p:
#         x = x.clone()
#         T = x.shape[-1]

#         shift = torch.randint(min_shift, max_shift + 1, (1,)).item()
#         if torch.rand(1).item() < 0.5:
#             shift = -shift

#         out = torch.zeros_like(x)

#         if shift > 0: # assign shifted part to zero tensor
#             out[..., shift:] = x[..., :T-shift]
#         else:
#             s = -shift
#             out[..., :T-s] = x[..., s:]
#         x = out
#         return x
#     else:
#         return x
    

# def partial_cqt(x, p=1.0, min_ratio=0.25, max_ratio=0.5):
#     if torch.rand(1).item() < p:
#         T = x.shape[-1]
#         keep_len = int(torch.empty(1).uniform_(min_ratio, max_ratio).item() * T) # keep min~max ratio
#         start = torch.randint(0, T - keep_len + 1, (1,)).item() # unmask start
#         out = torch.zeros_like(x)
#         out[..., start:start + keep_len] = x[..., start:start + keep_len]

#         return out
#     else:
#         return x
    

# def cyclic_subset(dataset, epoch, ratio=0.1):
#     N = len(dataset)
#     subset_size = int(N * ratio)

#     start = ((epoch - 1) * subset_size) % N
#     end = start + subset_size

#     if end <= N:
#         indices = list(range(start, end))
#     else:
#         indices = list(range(start, N)) + list(range(0, end - N))

#     return torch.utils.data.Subset(dataset, indices)
