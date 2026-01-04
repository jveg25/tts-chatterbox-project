import torch
import torchaudio
import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS

class ChatterboxInterface:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None

    def load_model(self):
        """Loads the ChatterboxTurboTTS model."""
        if self.model is None:
            # use from_pretrained as defined in chatterbox.tts_turbo
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        return self.model

    def generate_speech(self, text, reference_audio_path=None, exaggeration=0.0, cfg_weight=0.0, language="en"):
        """
        Generates speech using the chatterbox turbo model.
        
        Args:
            text (str): The text to synthesize.
            reference_audio_path (str, optional): Path to reference audio for voice cloning.
            exaggeration (float): expressiveness parameter (ignored by Turbo).
            cfg_weight (float): classifier-free guidance weight (ignored by Turbo).
            language (str): language code (Turbo is primarily English).
            
        Returns:
            tuple: (waveform: np.ndarray, sample_rate: int)
        """
        if self.model is None:
            self.load_model()

        kwargs = {
            "repetition_penalty": 1.2,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        
        if reference_audio_path:
            kwargs["audio_prompt_path"] = reference_audio_path
        
        # ChatterboxTurboTTS.generate is the method
        waveform = self.model.generate(text, **kwargs)

        # Ensure waveform is on CPU and converted to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
        return waveform, self.model.sr

if __name__ == "__main__":
    # Quick test
    interface = ChatterboxInterface()
    print(f"Device: {interface.device}")
