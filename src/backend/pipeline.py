import os
import uuid
import numpy as np
from .model_interface import ChatterboxInterface
from .text_processor import chunk_text
from .audio_utils import save_audio, stitch_audio

class TTSPipeline:
    def __init__(self):
        self.interface = ChatterboxInterface()
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_text(self, text, reference_audio_path=None, exaggeration=0.0, cfg_weight=0.0, language="en", output_format="wav", base_filename=None):
        """
        Processes text through the TTS pipeline.
        """
        # 1. Chunk text at punctuation marks
        chunks = chunk_text(text, max_length=250)
        
        waveforms = []
        sample_rate = 24000 
        
        # 2. Generate speech for each chunk and inject silence
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Generate the audio for the current chunk
            waveform, sr = self.interface.generate_speech(
                chunk,
                reference_audio_path=reference_audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language=language
            )
            
            # Make sure waveform is 1D
            if len(waveform.shape) > 1:
                waveform = waveform[0]
                
            waveforms.append(waveform)
            sample_rate = sr
            
            # 3. Inject silence if not the last chunk
            if idx < len(chunks) - 1:
                last_char = chunk.strip()[-1] if chunk.strip() else ""
                
                # Determine silence duration
                if last_char in ".!?":
                    pause_duration = 0.9  # 700ms for sentences
                elif last_char in ",;":
                    pause_duration = 0.6  # 400ms for commas/semicolons
                else:
                    pause_duration = 0.2  # 100ms for normal word gaps
                
                silence = np.zeros(int(sample_rate * pause_duration))
                waveforms.append(silence)
            
        # 4. Stitch audio
        if len(waveforms) > 0:
            final_waveform = stitch_audio(waveforms)
        else:
            return None
            
        # 4. Save to file
        if base_filename:
            filename = f"{base_filename}.{output_format}"
        else:
            filename = f"gen_{uuid.uuid4().hex[:8]}.{output_format}"
            
        output_path = os.path.join(self.output_dir, filename)
        save_audio(final_waveform, sample_rate, output_path, format=output_format)
        
        return output_path

    def process_batch(self, input_files, reference_audio_path=None, exaggeration=0.5, cfg_weight=0.5, language="en", output_format="wav"):
        """
        Processes multiple text files.
        """
        results = []
        for file_path in input_files:
            # Extract filename without extension
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            output_path = self.process_text(text, reference_audio_path, exaggeration, cfg_weight, language, output_format, base_filename=base_name)
            results.append((file_path, output_path))
        return results
