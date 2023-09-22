import os
import tempfile
import scipy.io.wavfile as wav
import ffmpeg
import cv2
from PIL import Image

import decord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, GaussianBlur, Grayscale, Resize
import torchaudio

decord.bridge.set_bridge('torch')
torchaudio.set_audio_backend("sox_io")


class AudioEncoder(nn.Module):
    """
    A PyTorch Module to encode audio data into a fixed-size vector 
    (also known as an "embedding"). This can be useful for various machine 
    learning tasks such as classification, similarity matching, etc.
    """
    def __init__(self, path):
        """
        Initialize the AudioEncoder object.

        Args:
            path (str): The file path where the pre-trained model is stored.
        """
        super().__init__()
        self.model = torch.jit.load(path)
        self.register_buffer('hidden', torch.zeros(2, 1, 256))

    def forward(self, audio):
        """
        The forward method is where the actual encoding happens. Given an 
        audio sample, this function returns its corresponding embedding.

        Args:
            audio (Tensor): A PyTorch tensor containing the audio data.
        
        Returns:
            Tensor: The embedding of the given audio.
        """
        self.reset()
        x = create_windowed_sequence(audio, 3200, cutting_stride=640, pad_samples=3200-640, cut_dim=1)
        embs = []                                           
        for i in range(x.shape[1]):
            emb, _, self.hidden = self.model(x[:, i], torch.LongTensor([3200]), init_state=self.hidden)
            embs.append(emb)
        return torch.vstack(embs)

    def reset(self):
        """
        Resets the hidden states in the model. Call this function 
        before processing a new audio sample to ensure that there is 
        no state carried over from the previous sample.
        """
        self.hidden = torch.zeros(2, 1, 256).to(self.hidden.device)


def get_audio_emb(audio_path, checkpoint, device):
    """
    This function takes the path of an audio file, loads it into a 
    PyTorch tensor, and returns its embedding.

    Args:
        audio_path (str): The file path of the audio to be loaded.
        checkpoint (str): The file path of the pre-trained model.
        device (str): The computing device ('cpu' or 'cuda').

    Returns:
        Tensor, Tensor: The original audio as a tensor and its corresponding embedding.
    """
    audio, audio_rate = torchaudio.load(audio_path, channels_first=False)
    assert audio_rate == 16000, 'Only 16 kHZ audio is supported.'
    audio = audio[None, None, :, 0].to(device)

    audio_encoder = AudioEncoder(checkpoint).to(device)

    emb = audio_encoder(audio)
    return audio, emb


def get_id_frame(path, random=False, resize=128):
    """
    Retrieves a frame from either a video or image file. This frame can 
    serve as an identifier or reference for the video or image.

    Args:
        path (str): File path to the video or image.
        random (bool): Whether to randomly select a frame from the video.
        resize (int): The dimensions to which the frame should be resized.

    Returns:
        Tensor: The image frame as a tensor.
    """
    if path.endswith('.mp4'):
        vr = decord.VideoReader(path)
        if random:
            idx = [np.random.randint(len(vr))]
        else:
            idx = [0]
        frame = vr.get_batch(idx).permute(0, 3, 1, 2)
    else:
        frame = load_image_to_torch(path).unsqueeze(0)
    
    frame = (frame / 255) * 2 - 1
    frame = Resize((resize, resize), antialias=True)(frame).float()
    return frame


def get_motion_transforms(args):
    """
    Applies a series of transformations like Gaussian blur and grayscale 
    conversion based on the provided arguments. This is commonly used for 
    data augmentation or preprocessing.

    Args:
        args (Namespace): Arguments containing options for motion transformations.
    
    Returns:
        Compose: A composed function of transforms.
    """
    motion_transforms = []
    if args.motion_blur:
        motion_transforms.append(GaussianBlur(5, sigma=2.0))
    if args.grayscale_motion:
        motion_transforms.append(Grayscale(1))
    return Compose(motion_transforms)


def save_audio(path, audio, audio_rate=16000):
    """
    Saves the audio data as a WAV file.

    Args:
        path (str): The file path where the audio will be saved.
        audio (Tensor or np.array): The audio data.
        audio_rate (int): The sampling rate of the audio, defaults to 16000Hz.
    """
    if torch.is_tensor(audio):
        aud = audio.squeeze().detach().cpu().numpy()
    else:
        aud = audio.copy()  # Make a copy so that we don't alter the object

    aud = ((2 ** 15) * aud).astype(np.int16)
    wav.write(path, audio_rate, aud)


def save_video(path, video, fps=25, scale=2, audio=None, audio_rate=16000, overlay_pts=None, ffmpeg_experimental=False):
    """
    Saves the video data as an MP4 file. Optionally includes audio and overlay points.

    Args:
        path (str): The file path where the video will be saved.
        video (Tensor or np.array): The video data.
        fps (int): Frames per second of the video.
        scale (int): Scaling factor for the video dimensions.
        audio (Tensor or np.array, optional): Audio data.
        audio_rate (int, optional): The sampling rate for the audio.
        overlay_pts (list of points, optional): Points to overlay on the video frames.
        ffmpeg_experimental (bool): Whether to use experimental ffmpeg options.

    Returns:
        bool: Success status.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    success = True    
    out_size = (scale * video.shape[-1], scale * video.shape[-2])
    video_path = get_temp_path(os.path.split(path)[0], ext=".mp4")
    if torch.is_tensor(video):
        vid = video.squeeze().detach().cpu().numpy()
    else:
        vid = video.copy()  # Make a copy so that we don't alter the object

    if np.min(vid) < 0:
        vid = 127 * vid + 127
    elif np.max(vid) <= 1:
        vid = 255 * vid

    is_color = True
    if vid.ndim == 3:
        is_color = False

    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), out_size, isColor=is_color)
    for i, frame in enumerate(vid):
        if is_color:
            frame = cv2.cvtColor(np.rollaxis(frame, 0, 3), cv2.COLOR_RGB2BGR)

        if scale != 1:
            frame = cv2.resize(frame, out_size)

        write_frame = frame.astype('uint8')

        if overlay_pts is not None:
            for pt in overlay_pts[i]:
                cv2.circle(write_frame, (int(scale * pt[0]), int(scale * pt[1])), 2, (0, 0, 0), -1)

        writer.write(write_frame)
    writer.release()

    inputs = [ffmpeg.input(video_path)['v']]

    if audio is not None:  # Save the audio file
        audio_path = swp_extension(video_path, ".wav")
        save_audio(audio_path, audio, audio_rate)
        inputs += [ffmpeg.input(audio_path)['a']]

    try:
        if ffmpeg_experimental:
            out = ffmpeg.output(*inputs, path, strict='-2', loglevel="panic", vcodec='h264').overwrite_output()
        else:
            out = ffmpeg.output(*inputs, path, loglevel="panic", vcodec='h264').overwrite_output()
        out.run(quiet=True)
    except:
        success = False

    if audio is not None and os.path.isfile(audio_path):
        os.remove(audio_path)
    if os.path.isfile(video_path):
        os.remove(video_path)

    return success


def load_image_to_torch(dir):
    """
    Load an image from disk and convert it to a PyTorch tensor.
    
    Args:
        dir (str): The directory path to the image file.
        
    Returns:
        torch.Tensor: A tensor representation of the image.
    """
    img = Image.open(dir).convert('RGB')
    img = np.array(img)
    return torch.from_numpy(img).permute(2, 0, 1)


def get_temp_path(tmp_dir, mode="", ext=""):
    """
    Generate a temporary file path for storing data.
    
    Args:
        tmp_dir (str): The directory where the temporary file will be created.
        mode (str, optional): A string to append to the file name.
        ext (str, optional): The file extension.
        
    Returns:
        str: The full path to the temporary file.
    """
    file_path = next(tempfile._get_candidate_names()) + mode + ext
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    file_path = os.path.join(tmp_dir, file_path)
    return file_path


def swp_extension(file, ext):
    """
    Swap the extension of a given file name.
    
    Args:
        file (str): The original file name.
        ext (str): The new extension.
        
    Returns:
        str: The file name with the new extension.
    """
    return os.path.splitext(file)[0] + ext


def pad_both_ends(tensor, left, right, dim=0):
    """
    Pad a tensor on both ends along a specific dimension.
    
    Args:
        tensor (torch.Tensor): The tensor to be padded.
        left (int): The padding size for the left side.
        right (int): The padding size for the right side.
        dim (int, optional): The dimension along which to pad.
        
    Returns:
        torch.Tensor: The padded tensor.
    """
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims - 1

    padding = [0] * 2 * no_dims
    padding[2 * (no_dims - dim - 1)] = left
    padding[2 * (no_dims - dim - 1) + 1] = right
    return F.pad(tensor, padding, "constant", 0)


def cut_n_stack(seq, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    """
    Divide a sequence tensor into smaller snips and stack them.
    
    Args:
        seq (torch.Tensor): The original sequence tensor.
        snip_length (int): The length of each snip.
        cut_dim (int, optional): The dimension along which to cut.
        cutting_stride (int, optional): The stride length for cutting. Defaults to snip_length.
        pad_samples (int, optional): Number of samples to pad at both ends.
        
    Returns:
        torch.Tensor: A tensor containing the stacked snips.
    """
    if cutting_stride is None:
        cutting_stride = snip_length

    pad_left = pad_samples // 2
    pad_right = pad_samples - pad_samples // 2

    seq = pad_both_ends(seq, pad_left, pad_right, dim=cut_dim)

    stacked = seq.narrow(cut_dim, 0, snip_length).unsqueeze(0)
    iterations = (seq.size()[cut_dim] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(cut_dim, i * cutting_stride, snip_length).unsqueeze(0)))
    return stacked


def create_windowed_sequence(seqs, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    """
    Create a windowed sequence from a list of sequences.
    
    Args:
        seqs (list of torch.Tensor): List of sequence tensors.
        snip_length (int): The length of each snip.
        cut_dim (int, optional): The dimension along which to cut.
        cutting_stride (int, optional): The stride length for cutting. Defaults to snip_length.
        pad_samples (int, optional): Number of samples to pad at both ends.
        
    Returns:
        torch.Tensor: A tensor containing the windowed sequences.
    """
    windowed_seqs = []
    for seq in seqs:
        windowed_seqs.append(cut_n_stack(seq, snip_length, cut_dim, cutting_stride, pad_samples).unsqueeze(0))

    return torch.cat(windowed_seqs)