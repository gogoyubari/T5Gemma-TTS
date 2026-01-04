"""
SRT-based Voice Dubbing Script for T5Gemma-TTS.

Replaces Japanese speech with English speech at matching timestamps,
using voice cloning from the original Japanese audio.

Usage:
    python srt_dubbing.py \
        --ja_srt inputs/0305_ja.srt \
        --en_srt inputs/0305_en.srt \
        --input_wav inputs/0305_vocals.wav \
        --output_wav outputs/0305_dubbed.wav
"""

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from data.tokenizer import AudioTokenizer
from inference_tts_utils import (
    inference_one_sample,
    normalize_text_with_lang,
    transcribe_audio,
)
from similarity_and_distance import find_similarity_and_distance
from compare_audio_features import compare_with_reference_set

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# SRT Parser
# ---------------------------------------------------------------------------

@dataclass
class SRTSegment:
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str


def parse_timestamp(ts: str) -> float:
    """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
    # Handle both comma and period as decimal separator
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt(filepath: str) -> List[SRTSegment]:
    """Parse SRT file and return list of segments."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    # Split by double newline (segment separator)
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # First line: index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamps
        timestamp_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip()
        )
        if not timestamp_match:
            continue

        start_time = parse_timestamp(timestamp_match.group(1))
        end_time = parse_timestamp(timestamp_match.group(2))

        # Remaining lines: text
        text = '\n'.join(lines[2:]).strip()

        segments.append(SRTSegment(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text
        ))

    return segments


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_resources(
    model_dir: str,
    xcodec2_model_name: str,
    xcodec2_sample_rate: int,
    cpu_codec: bool = False,
    cpu_whisper: bool = False,
):
    """Load TTS model and tokenizers."""
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise ImportError("Please install transformers.")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[Info] Loading model from {model_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Check device map
    has_device_map = bool(getattr(model, "hf_device_map", None))
    if has_device_map:
        for dev in model.hf_device_map.values():
            if dev not in ("cpu", "disk"):
                if isinstance(dev, int):
                    device = f"cuda:{dev}"
                elif isinstance(dev, torch.device):
                    device = str(dev)
                elif isinstance(dev, str):
                    device = dev
                break

    model.eval()
    cfg = model.config

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"[Info] Loading XCodec2 from {xcodec2_model_name}...")
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        device=torch.device("cpu") if cpu_codec else torch.device(device),
        model_name=xcodec2_model_name,
        sample_rate=xcodec2_sample_rate,
    )

    codec_audio_sr = xcodec2_sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    # Whisper device
    if cpu_whisper:
        whisper_device = "cpu"
    elif torch.cuda.is_available():
        whisper_device = "cuda"
    elif torch.backends.mps.is_available():
        whisper_device = "mps"
    else:
        whisper_device = "cpu"

    return {
        "model": model,
        "cfg": cfg,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": audio_tokenizer,
        "device": device,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "whisper_device": whisper_device,
    }


# ---------------------------------------------------------------------------
# Audio Extraction
# ---------------------------------------------------------------------------

def extract_audio_segment(
    input_wav: str,
    start_time: float,
    end_time: float,
    output_path: str,
    target_sr: int = 16000,
):
    """Extract audio segment from WAV file and save to output_path."""
    # Load audio
    info = sf.info(input_wav)
    sr = info.samplerate

    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)

    wav, _ = sf.read(input_wav, start=start_frame, stop=end_frame, dtype='float32')

    # Convert to mono if stereo
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.transforms.Resample(sr, target_sr)(wav_tensor)
        wav = wav_tensor.squeeze(0).numpy()

    # Save
    sf.write(output_path, wav, target_sr, subtype='PCM_16')
    return output_path


def select_best_sample(
    samples: List[np.ndarray],
    target_text: str,
    reference_audio_path: str,
    sample_rate: int,
    whisper_device: str = "cuda",
    tmpdir: str = "/tmp",
    top_k_voice: int = 5,
) -> tuple:
    """
    複数の生成サンプルから最適なものを選択する（2段階フィルタリング）。

    選択基準:
    1. MFCC距離で声質が近い上位 top_k_voice 本を選択
    2. Whisper で逆起こし → target_text との Levenshtein 類似度でベストを選択

    Returns:
        (best_sample, best_index, best_transcript, best_similarity, best_rms)
    """
    # Stage 1: Save all samples and calculate MFCC distance
    print(f"    [Stage 1] Calculating MFCC distance for {len(samples)} samples...")
    voice_candidates = []

    for i, sample in enumerate(samples):
        tmp_path = os.path.join(tmpdir, f"sample_{i}.wav")
        sf.write(tmp_path, sample, sample_rate, subtype='PCM_16')

        try:
            mfcc_distance = compare_with_reference_set(reference_audio_path, tmp_path)
        except Exception as e:
            print(f"    [Warning] MFCC comparison failed for sample {i}: {e}")
            mfcc_distance = float('inf')

        rms = float(np.sqrt(np.mean(sample ** 2)))

        voice_candidates.append({
            "index": i,
            "sample": sample,
            "tmp_path": tmp_path,
            "mfcc_distance": mfcc_distance,
            "rms": rms,
        })

        print(f"    Sample {i+1}: mfcc_distance={mfcc_distance:.4f}, rms={rms:.4f}")

    # Sort by MFCC distance (ascending) and select top_k_voice
    voice_candidates.sort(key=lambda c: c["mfcc_distance"])
    top_voice_candidates = voice_candidates[:top_k_voice]

    print(f"    -> Top {top_k_voice} by voice quality: {[c['index']+1 for c in top_voice_candidates]}")

    # Stage 2: Transcribe top candidates and calculate Levenshtein similarity
    print(f"    [Stage 2] Transcribing top {len(top_voice_candidates)} samples...")
    text_candidates = []

    for c in top_voice_candidates:
        try:
            transcript = transcribe_audio(c["tmp_path"], whisper_device)
        except Exception as e:
            print(f"    [Warning] Transcribe failed for sample {c['index']}: {e}")
            transcript = ""

        similarity, _ = find_similarity_and_distance(target_text, transcript)

        text_candidates.append({
            **c,
            "transcript": transcript,
            "similarity": similarity,
        })

        print(f"    Sample {c['index']+1}: similarity={similarity:.3f}")
        print(f"             transcript: {transcript[:60]}...")

    # Sort by Levenshtein similarity (descending)
    text_candidates.sort(key=lambda c: -c["similarity"])

    best = text_candidates[0]
    print(f"    -> Selected sample {best['index']+1} (mfcc={best['mfcc_distance']:.4f}, similarity={best['similarity']:.3f})")

    return (
        best["sample"],
        best["index"],
        best["transcript"],
        best["similarity"],
        best["rms"],
    )


# ---------------------------------------------------------------------------
# TTS Inference
# ---------------------------------------------------------------------------

def run_tts_inference(
    resources: dict,
    reference_speech: str,
    reference_text: str,
    target_text: str,
    target_duration: float,
    num_samples: int = 5,
) -> List[np.ndarray]:
    """Run TTS inference and return list of generated waveforms."""
    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    # Normalize text
    target_text_norm, lang_code = normalize_text_with_lang(target_text, "en")
    reference_text_norm, _ = normalize_text_with_lang(reference_text, "ja")

    # Get reference audio info for prompt_end_frame
    info = sf.info(reference_speech)
    prompt_end_frame = int(info.frames)

    decode_config = {
        "top_k": 30,
        "top_p": 0.9,
        "min_p": 0.0,
        "temperature": 0.8,
        "stop_repetition": 3,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": [],
        "sample_batch_size": 1,
    }

    concat_audio, gen_audio = inference_one_sample(
        model=model,
        model_args=cfg,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=reference_speech,
        target_text=target_text_norm,
        lang=lang_code,
        device=device,
        decode_config=decode_config,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_duration,
        prefix_transcript=reference_text_norm,
        multi_trial=[],
        repeat_prompt=0,
        return_frames=False,
        num_samples=num_samples,
    )

    # Convert to numpy arrays
    samples = []
    if num_samples == 1:
        # Single sample case
        audio = gen_audio[0].detach().cpu()
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        samples.append(audio.numpy())
    else:
        # Multiple samples case
        for i, audio in enumerate(gen_audio):
            wav = audio[0].detach().cpu()
            if wav.ndim == 2 and wav.shape[0] == 1:
                wav = wav.squeeze(0)
            samples.append(wav.numpy())

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRT-based Voice Dubbing with T5Gemma-TTS")
    parser.add_argument("--reference_srt", type=str, required=True, help="Reference SRT file path")
    parser.add_argument("--reference_voice", type=str, required=True, help="Reference WAV path")
    parser.add_argument("--target_srt", type=str, required=True, help="Target SRT file path")
    parser.add_argument("--output_voice", type=str, default=None, help="Output WAV path (default: <input_basename>_dubbed.wav)")
    parser.add_argument("--model_dir", type=str, default="t5gemma_voice_hf", help="TTS model directory")
    parser.add_argument("--xcodec2_model_name", type=str, default="HKUSTAudio/xcodec2", help="XCodec2 model name")
    parser.add_argument("--xcodec2_sample_rate", type=int, default=16000, help="XCodec2 sample rate")
    parser.add_argument("--cpu_codec", action="store_true", help="Run XCodec2 on CPU")
    parser.add_argument("--cpu_whisper", action="store_true", help="Run Whisper on CPU")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per segment")
    args = parser.parse_args()

    # Set default output_wav if not provided
    if args.output_voice is None:
        base = os.path.splitext(args.reference_voice)[0]
        args.output_voice = f"{base}_dubbed.wav"

    # Parse SRT files
    print("[Info] Parsing SRT files...")
    ref_segments = parse_srt(args.reference_srt)
    tgt_segments = parse_srt(args.target_srt)

    print(f"[Info] Found {len(ref_segments)} Reference segments, {len(tgt_segments)} Target segments")

    if len(ref_segments) != len(tgt_segments):
        print(f"[Warning] Segment count mismatch! Using minimum: {min(len(ref_segments), len(tgt_segments))}")

    num_segments = min(len(ref_segments), len(tgt_segments))

    # Load model
    resources = load_resources(
        model_dir=args.model_dir,
        xcodec2_model_name=args.xcodec2_model_name,
        xcodec2_sample_rate=args.xcodec2_sample_rate,
        cpu_codec=args.cpu_codec,
        cpu_whisper=args.cpu_whisper,
    )

    output_sr = args.xcodec2_sample_rate  # 16000

    # Calculate total duration
    total_duration = max(seg.end_time for seg in ref_segments[:num_segments])
    print(f"[Info] Total duration: {total_duration:.2f} seconds")

    # Create output buffer (silence)
    output_samples = int(total_duration * output_sr)
    output_buffer = np.zeros(output_samples, dtype=np.float32)

    # Create temp directory for reference audio segments
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_segments):
            ref_seg = ref_segments[i]
            tgt_seg = tgt_segments[i]

            print(f"\n[Info] Processing segment {i+1}/{num_segments}")
            print(f"  Time: {ref_seg.start_time:.2f}s - {ref_seg.end_time:.2f}s")
            print(f"  REF: {ref_seg.text[:50]}...")
            print(f"  TGT: {tgt_seg.text[:50]}...")

            target_duration = ref_seg.end_time - ref_seg.start_time

            # Extract reference audio segment
            ref_audio_path = os.path.join(tmpdir, f"ref_{i:04d}.wav")
            extract_audio_segment(
                input_wav=args.reference_voice,
                start_time=ref_seg.start_time,
                end_time=ref_seg.end_time,
                output_path=ref_audio_path,
                target_sr=output_sr,
            )

            # Run TTS (generate multiple samples)
            try:
                generated_samples = run_tts_inference(
                    resources=resources,
                    reference_speech=ref_audio_path,
                    reference_text=ref_seg.text,
                    target_text=tgt_seg.text,
                    target_duration=target_duration,
                    num_samples=args.num_samples,
                )

                # Select best sample: MFCC (top 5) -> Levenshtein (best 1)
                print(f"  Selecting best from {len(generated_samples)} samples...")
                best_audio, best_idx, best_transcript, best_similarity, best_rms = select_best_sample(
                    samples=generated_samples,
                    target_text=tgt_seg.text,
                    reference_audio_path=ref_audio_path,
                    sample_rate=output_sr,
                    whisper_device=resources["whisper_device"],
                    tmpdir=tmpdir,
                )

                # Place in output buffer
                start_sample = int(ref_seg.start_time * output_sr)
                end_sample = start_sample + len(best_audio)

                # Handle overflow
                if end_sample > len(output_buffer):
                    best_audio = best_audio[:len(output_buffer) - start_sample]
                    end_sample = len(output_buffer)

                output_buffer[start_sample:start_sample + len(best_audio)] = best_audio

                print(f"  Generated: {len(best_audio)} samples ({len(best_audio)/output_sr:.2f}s)")

            except Exception as e:
                print(f"  [Error] Failed to generate: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save output
    output_dir = os.path.dirname(args.output_voice)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sf.write(args.output_voice, output_buffer, output_sr, subtype='PCM_16')
    print(f"\n[Info] Saved output to {args.output_voice}")
    print(f"[Info] Output: {output_sr}Hz, 16bit, mono, {len(output_buffer)/output_sr:.2f}s")


if __name__ == "__main__":
    main()
