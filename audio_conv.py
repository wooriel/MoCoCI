import math
import numpy as np
import pretty_midi
from pydub import AudioSegment
import librosa
import essentia.standard as es
import h5py, json, csv


def from_midi(file_path, sample_rate=32000):
    midi = pretty_midi.PrettyMIDI(file_path)
    wav_amp = midi.synthesize(fs=sample_rate).astype(np.float32)

    return wav_amp, sample_rate


def get_pcm_samples(audio):
    """_summary_: computes wave_amplitude function and sample rate

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
    Returns:
        pcm_float (float ndarray): x: sample n | y: amplitude (in PCM float in range [-1.0, 1.0)] )
        sample rate (int): n samples / second
    """
    if audio.channels == 1: # mono
        audio.set_channels(1)

    samples = np.asarray(audio.get_array_of_samples()) # raw audio -> numeric samples, x: sample n | y: audio sample
    bits_sample = audio.sample_width * 8 # 16
    sample_rate = audio.frame_rate # sample rate: n samples / second
    pcm_float = samples.astype(np.float32) / (2 ** (bits_sample - 1)) # signed sample-bit PCM(pulse code modulation) / precision: 32 bit floating point

    return pcm_float, sample_rate


def get_frame_params(sample_rate):
    """_summary_

    Args:
        sample_rate (int): number of samples in one second
    Returns:
        frame_size (int): size of single frame
        hop_length (int): size of hop length to make 50% overlap
    """
    frame_size = round(0.096 * sample_rate)  # 96 ms
    hop_length = round(0.048 * sample_rate)  # 48 ms
    return frame_size, hop_length


def get_num_frames(music_duration, sample_rate, frame_size, hop_length, center):
    """_summary_: given setting, calculate number of frames

    Args:
        music_duration (float): music duration in seconds
        sample_rate (int): number of sample in one second
        frame_size (int): single frame size
        hop_length (int): space between starting point between two frames
        center (int): if center = True, hop_length * current frame index + (frame_size // 2) would be the center
            referred to librosa component / center=True corresponds to essentia startFromZero=False
    """
    total_samples = math.floor(music_duration * sample_rate)
    if total_samples < frame_size:
        return 1
    
    if center == True:
        pad = frame_size // 2
        num_samples += 2 * pad

    return (num_samples - frame_size) // hop_length + 1


def get_croped_chroma(wav_amp, chroma, frame_length, hop_length, threshold=-60.0): # used on chromagram
    """_summary_: cuts start and end chroma is the db is lower than the shreshold

    Args:
        audio (pydub.AudioSegment object): in-memory PCM audio representation
        chroma (np.ndarray): (n_chroma, n_frames) size chromagram
        frame_length (int): RMS analysis frame length in samples (rms_frame_length for CQT)
        hop_length (int): shift distance per frame
        threshold (float, def=-60.0): silence threshold relative to maximum RMS, in dB
    Returns:
        chroma (np.ndarray): cropped chromagram
    """
    rms = librosa.feature.rms(y=wav_amp, frame_length=frame_length, hop_length=hop_length, center=False)[0] # root mean square
    rms_db = librosa.amplitude_to_db(rms, ref=np.max,)
    act_frames = min(chroma.shape[1], len(rms_db)) # exact end

    chroma = chroma[:, :act_frames].copy()
    rms_db = rms_db[:act_frames]

    active = rms_db >= threshold
    if np.any(active):
        start = np.argmax(active)
        end = len(active) - np.argmax(active[::-1])
        chroma = chroma[:, start:end]

    return chroma


def get_cqt_spec(file_path, format="mp3", bins_per_octave=12):
    """_summary_: waveform > cqt > magnitude/power spectrum
        f_k = f_min 2^(k/B) (B = bins per octave) < bin center spacing
    
    Args:
        file_path (path object): path to raw audio file (mp3)
        hop_length (int): shift distance per frame
        bins_per_octave (int): how many frequency bins per octave
    Returns:
        cqt_chroma (samp_rate/hop_length, n_frames): x: CQT frequency | y: time in num_frames
        20 * log_10 (A/A_ref), reference amplitude value is the maximum value on the window
    """
    if format == "mp3":
        audio = AudioSegment.from_mp3(file_path)
        wav_amp, sample_rate = get_pcm_samples(audio)
    elif format == "mid":
        wav_amp, sample_rate = from_midi(file_path)# function

    # sample_rate = audio.frame_rate
    frame_size, hop_length = get_frame_params(sample_rate)
    n_bins = bins_per_octave * 8 # (num_octaves) = n_bins / bins_per_octave

    cqt_spec = librosa.cqt(wav_amp, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave,)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max,)

    return cqt_db


def get_hpcp(file_path, format="mp3", hpcp_size=12, minFreq=40, maxFreq=5000, crop=False):
    """_summary_: waveform > FFT > magnitude/power spectrum > spectral peak detection > frequencies + magnitudes > HPCP weighting > HPCP

    Args:
        file_path (path object): path to raw audio file (mp3)
        frame_size (int, def=4096): frame window size
        hop_length (int, def=512): shift distance per frame
        hpcp_size (int, def=12): number of bin for hpcp
    Returns:
        cqt_chroma (12, n_frames): x: CQT frequency in 12 pitch class bins | y: time in num_frames
            Constant-Q transform (CQT) converts frequency in logarithmic
            semitone has constant ratio (e.g. A4 = 440Hz, A5 = 880Hz)
            follow logarithmic, pitch-oriented spacing
    """
    if format == "mp3":
        audio = AudioSegment.from_mp3(file_path)
        wav_amp, sample_rate = get_pcm_samples(audio)
    elif format == "mid":
        wav_amp, sample_rate = from_midi(file_path)# function

    frame_size, hop_length = get_frame_params(sample_rate)

    window = es.Windowing(type="blackmanharris62") # smooth weighting function object
    spectrum = es.Spectrum(size=frame_size) # FFT transform (with abs) object

    spectral_peaks = es.SpectralPeaks(
        sampleRate=sample_rate,
        orderBy="magnitude",
        magnitudeThreshold=0,
        minFrequency=minFreq,
        maxFrequency=min(maxFreq, sample_rate / 2.0),
        maxPeaks=60,
    ) # later make this i 
    # magnitudeThreshold=1e-5

    hpcp_algorithm = es.HPCP(
        size=hpcp_size, sampleRate=sample_rate,
        minFrequency=minFreq, maxFrequency=min(maxFreq, sample_rate / 2.0), referenceFrequency=440.0,
        weightType="squaredCosine",
        windowSize=4.0/3.0,
        harmonics=8,
        normalized="unitMax",
    ) # minFrequency .. is the baseline reference setting
    hpcp_frames = []

    for frame in es.FrameGenerator(wav_amp, frameSize=frame_size, hopSize=hop_length, startFromZero=True, lastFrameToEndOfFile=False):
        windowed = window(frame)
        spec = spectrum(windowed)
        freq, mag = spectral_peaks(spec)
        hpcp = hpcp_algorithm(freq, mag,)
        hpcp_frames.append(hpcp)

    hpcp = np.asarray(hpcp_frames, dtype=np.float32).T
    if crop:
        crop_hpcp = get_croped_chroma(wav_amp, hpcp, frame_length=frame_size, hop_length=hop_length)
        return crop_hpcp
    else:
        return hpcp


def load_json_metadata(file_path):
    with open(file_path, "r") as f:
        data = json.load(f) # nested dictionary of original work and performance

        return data

def load_single_h5py(file_path, key="hpcp", transpose=True):
    with h5py.File(file_path, "r") as f:
        hpcp = np.array(f[key], dtype=np.float32)

    if transpose and hpcp.ndim == 2:
        if hpcp.shape[0] > hpcp.shape[1]:
            hpcp = hpcp.T

    return hpcp


def write_csv(file_path, rows):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)