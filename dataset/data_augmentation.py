import torch


# used on all dataset
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


# used on pndatacosdataset
def shift_pitch_hpcp(x, p=1.0, max_pitch_shift=6):
    if torch.rand(1).item() < p:
        shift = torch.randint(-max_pitch_shift, max_pitch_shift + 1, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=0)
        return x
    else:
        return x


# used on pndatacosdataset
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
    

# used on pndatacosdataset
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


# can later be applied to pndatacosdataset
def splice_same_song_hpcp(x, p=0.2, seg_len=500): # concat or replacing temportal regions of the same song
    if torch.rand(1).item() < p:
        T = x.shape[-1]

        if T < 2 * seg_len:
            return x

        s1 = torch.randint(0, T - seg_len + 1, (1,)).item()
        s2 = torch.randint(0, T - seg_len + 1, (1,)).item()

        x_aug = x.clone()
        x_aug[..., s1:s1 + seg_len] = x[..., s2:s2 + seg_len]

        return x_aug
    else:
        return x


# used on ci_copyright, ci_mcic
def cut_hpcp(x, p=1.0, ratio=0.25):
    if torch.rand(1).item() < p:
        T = x.shape[-1]
        # keep_len = int(torch.empty(1).uniform_(ratio).item() * T) # keep min~max ratio
        start = torch.randint(0, T - ratio + 1, (1,)).item() # unmask start
        out = x[..., start:start + ratio]

        return out
    else:
        return x