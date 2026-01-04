import gradio as gr
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.pipeline import TTSPipeline

pipeline = TTSPipeline()

def tts_interface(text, file_upload, reference_audio, exaggeration, cfg_weight, language, output_format):
    input_text = ""
    base_name = None
    if text:
        input_text = text
    elif file_upload is not None:
        base_name = os.path.splitext(os.path.basename(file_upload.name))[0]
        with open(file_upload.name, 'r', encoding='utf-8') as f:
            input_text = f.read()
    
    if not input_text:
        return None, "Please provide text or upload a file."
    
    # reference_audio can be a path (string) or a dict from Gradio's Audio component
    ref_path = None
    if reference_audio is not None:
        if isinstance(reference_audio, str):
            ref_path = reference_audio
        elif isinstance(reference_audio, tuple): # (sample_rate, data)
            # We need to save it to a temp file first if it's a recording
            import soundfile as sf
            import tempfile
            sr, data = reference_audio
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, data, sr)
            ref_path = temp_file.name
        else:
            # Check if it's a file object from upload
            ref_path = reference_audio
            
    try:
        output_path = pipeline.process_text(
            input_text,
            reference_audio_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language=language,
            output_format=output_format,
            base_filename=base_name
        )
        return output_path, f"Successfully generated: {os.path.basename(output_path)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def batch_interface(files, reference_audio, exaggeration, cfg_weight, language, output_format):
    if not files:
        return None, "Please upload files."
    
    ref_path = None
    if reference_audio is not None:
        if isinstance(reference_audio, str):
            ref_path = reference_audio
        elif isinstance(reference_audio, tuple):
            import soundfile as sf
            import tempfile
            sr, data = reference_audio
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, data, sr)
            ref_path = temp_file.name
        else:
            ref_path = reference_audio

    file_paths = [f.name for f in files]
    
    try:
        results = pipeline.process_batch(
            file_paths,
            reference_audio_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language=language,
            output_format=output_format
        )
        
        output_files = [res[1] for res in results]
        return output_files, f"Successfully generated {len(output_files)} files."
    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Chatterbox TTS Studio") as demo:
    gr.Markdown("# 🎙️ Chatterbox TTS Studio")
    gr.Markdown("Create high-quality speech using Resemble AI's Chatterbox-Turbo model.")
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Single Text"):
                    text_input = gr.Textbox(label="Input Text", lines=5, placeholder="Enter text here...")
                    file_input = gr.File(label="Or Upload .txt File", file_types=[".txt"])
                
                with gr.TabItem("Batch Processing"):
                    batch_inputs = gr.File(label="Upload multiple .txt Files", file_count="multiple", file_types=[".txt"])
            
            with gr.Accordion("Voice Cloning (Optional)", open=True):
                ref_audio = gr.Audio(label="Reference Audio (Upload or Record)", type="filepath")
            
            with gr.Accordion("Parameters", open=False):
                exagg = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Exaggeration")
                cfg = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="CFG / Pacing")
                lang = gr.Dropdown(choices=["en", "es", "fr", "de", "it", "jp", "zh"], value="en", label="Language")
                fmt = gr.Radio(choices=["wav", "mp3"], value="wav", label="Output Format")
                
            generate_btn = gr.Button("Generate Speech", variant="primary")
            batch_btn = gr.Button("Process Batch", variant="secondary")

        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
            batch_output_files = gr.File(label="Generated Batch Files")
            status_msg = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=tts_interface,
        inputs=[text_input, file_input, ref_audio, exagg, cfg, lang, fmt],
        outputs=[output_audio, status_msg]
    )
    
    batch_btn.click(
        fn=batch_interface,
        inputs=[batch_inputs, ref_audio, exagg, cfg, lang, fmt],
        outputs=[batch_output_files, status_msg]
    )

if __name__ == "__main__":
    demo.launch()
