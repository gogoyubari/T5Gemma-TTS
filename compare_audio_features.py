import os
import numpy as np
import librosa

def audio_to_mfcc_feature(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.average(mfcc, axis=1)  # 時間平均を取る
    mfcc = mfcc.flatten() # 1次元配列に変換
    mfcc = mfcc.tolist()
    mfcc = mfcc[1:13]  # 低次の係数を取り出す（12次まで取り出すことが多い）
    
    return mfcc

def compare_with_reference_set(reference_file, target_file):
    ref_feature = audio_to_mfcc_feature(reference_file)
    target_feature = audio_to_mfcc_feature(target_file)
    
    ref_np = np.array(ref_feature)
    tgt_np = np.array(target_feature)
    distance = np.linalg.norm(ref_np - tgt_np)

    return distance

if __name__ == "__main__":
    reference_dir = "./reference_wav_files"
    target_dir = "./target_wav_files"
    reference_files = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir) if f.lower().endswith('.wav')]
    target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.wav')]

    for reference_file in reference_files:
        for target_file in target_files: 
                   
            distance = compare_with_reference_set(reference_file, target_file)
            print(f"{reference_file} {target_file} {distance:.4f}")