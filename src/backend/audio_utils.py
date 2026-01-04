import os
import soundfile as sf
from pydub import AudioSegment
import numpy as np

def save_audio(waveform, sample_rate, output_path, format='wav'):
    """
    Saves audio to disk in requested format.
    
    Args:
        waveform (np.ndarray): The audio waveform.
        sample_rate (int): Sample rate of the audio.
        output_path (str): Path where the file will be saved.
        format (str): 'wav' or 'mp3'.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If waveform is 2D (batch, samples), take the first one
    if len(waveform.shape) > 1:
        waveform = waveform[0]

    if format.lower() == 'wav':
        sf.write(output_path, waveform, sample_rate)
    elif format.lower() == 'mp3':
        # Convert to int16 for pydub
        # Waveform from chatterbox is likely in range [-1, 1]
        waveform_int16 = (waveform * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=waveform_int16.dtype.itemsize,
            channels=1
        )
        audio_segment.export(output_path, format="mp3")
    else:
        raise ValueError(f"Unsupported format: {format}")

def stitch_audio(waveforms):
    """
    Stitches multiple waveforms together.
    
    Args:
        waveforms (list): List of numpy arrays.
        
    Returns:
        np.ndarray: Concatenated waveform.
    """
    if not waveforms:
        return np.array([])
    
    # Normalize shapes - ensure they are 1D
    flat_waveforms = []
    for w in waveforms:
        if len(w.shape) > 1:
            flat_waveforms.append(w[0])
        else:
            flat_waveforms.append(w)
            
    return np.concatenate(flat_waveforms)
