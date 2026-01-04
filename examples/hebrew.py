"""
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav
wget https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.int8.onnx

uv sync
uv pip install dicta-onnx soundfile

Optional (e.g. on DGX Spark):
uv pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

uv run --no-sync examples/hebrew.py
"""


from dicta_onnx import Dicta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.utils import get_device
import soundfile as sf
import numpy as np


def add_diacritics(sentence: str) -> str:
    dicta = Dicta('./dicta-1.0.int8.onnx')
    return dicta.add_diacritics(sentence)


def create_audio(text: str, ref: str) -> tuple[np.ndarray, int]:
    device = get_device()
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    wav = model.generate(text, language_id="he", audio_prompt_path=ref)
    return (wav.squeeze(0).numpy(), model.sr)


def main():
    text = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    audio_path = "audio.wav"
    ref = "male1.wav"

    with_diacritics = add_diacritics(text)
    samples, sample_rate = create_audio(with_diacritics, ref)
    sf.write(audio_path, samples, sample_rate)
    print(f'Created {audio_path}')

if __name__ == "__main__":
    main()