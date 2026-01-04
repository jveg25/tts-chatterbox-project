import torch
import torchaudio
import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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
        
        self.turbo_model = None
        self.mtl_model = None

    def load_turbo_model(self):
        """Loads the ChatterboxTurboTTS model (Optimized for English)."""
        if self.turbo_model is None:
            print("Loading Chatterbox Turbo model...")
            self.turbo_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        return self.turbo_model

    def load_mtl_model(self):
        """Loads the ChatterboxMultilingualTTS model (Supports 23+ languages)."""
        if self.mtl_model is None:
            print("Loading Chatterbox Multilingual model...")
            
            # Monkey-patch torch.load to handle map_location for the library
            # as it sometimes forgets to pass it, causing CUDA errors on Mac.
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = 'cpu'
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_torch_load
            try:
                self.mtl_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
                
        return self.mtl_model

    def generate_speech(self, text, reference_audio_path=None, exaggeration=0.0, cfg_weight=0.0, language="en"):
        """
        Generates speech switching between Turbo and Multilingual models.
        
        Args:
            text (str): The text to synthesize.
            reference_audio_path (str, optional): Path to reference audio for voice cloning.
            exaggeration (float): Expressiveness/Emotion (0.0 to 1.0).
            cfg_weight (float): CFG weight (0.0 to 1.0).
            language (str): language code (e.g., 'en', 'es', 'fr').
            
        Returns:
            tuple: (waveform: np.ndarray, sample_rate: int)
        """
        is_english = (language.lower() == "en")
        
        if is_english:
            model = self.load_turbo_model()
            kwargs = {
                "repetition_penalty": 1.2,
                "temperature": 0.8,
                "top_p": 0.95,
            }
        else:
            model = self.load_mtl_model()
            kwargs = {
                "language_id": language.lower(),
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": 0.8,
                "repetition_penalty": 2.0,
            }
        
        if reference_audio_path:
            kwargs["audio_prompt_path"] = reference_audio_path
        
        # Generate the audio
        waveform = model.generate(text, **kwargs)

        # Ensure waveform is on CPU and converted to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
        return waveform, model.sr

if __name__ == "__main__":
    # Quick test
    interface = ChatterboxInterface()
    print(f"Device: {interface.device}")
