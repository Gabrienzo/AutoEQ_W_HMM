import time
import sys
import numpy as np
import librosa
import soundcard as sc
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"

PRESET_MUSIC = os.path.abspath("presets/music_preset.txt")
PRESET_PODCAST = os.path.abspath("presets/podcast_preset.txt")

SAMPLE_RATE = 22050
DURATION = 2.0 

def get_system_loopback():
    default_speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
    return mic

def extract_features_from_buffer(data, sr):
    y = np.mean(data, axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def update_apo_config(mode):
    try:
        preset_path = PRESET_MUSIC if mode == 'music' else PRESET_PODCAST
        line_to_write = f"Include: {preset_path}"
        
        current_content = ""
        if os.path.exists(APO_CONFIG_PATH):
            with open(APO_CONFIG_PATH, 'r') as f:
                current_content = f.read().strip()
        
        if current_content != line_to_write:
            with open(APO_CONFIG_PATH, 'w') as f:
                f.write(line_to_write)
            return True
        return False
            
    except Exception as e:
        print(f"\nErro ao atualizar Equalizer APO: {e}")
        return False

def print_debug_status(score_music, score_podcast, active_mode):
    # Log-likelihoods sao numeros negativos grandes (ex: -5000). 
    # Para visualizacao, normalizamos a diferenca.
    
    diff = abs(score_music - score_podcast)
    max_len = 30
    
    # Define quem esta ganhando para desenhar a barra
    if score_music > score_podcast:
        confidence = min(diff / 500, 1.0) # 500 pontos de diferenca = 100% barra
        bar_len = int(confidence * max_len)
        bar_music = "█" * bar_len + "-" * (max_len - bar_len)
        bar_podcast = "-" * max_len
        winner = "MUSICA"
    else:
        confidence = min(diff / 500, 1.0)
        bar_len = int(confidence * max_len)
        bar_podcast = "█" * bar_len + "-" * (max_len - bar_len)
        bar_music = "-" * max_len
        winner = "PODCAST"

    # Cores ANSI para terminal (se suportado pelo Windows Terminal/CMD)
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    status_color = GREEN if winner == active_mode.upper() else YELLOW
    
    sys.stdout.write(f"\r{RESET}[ {status_color}{winner}{RESET} ] | "
                     f"MUS: [{bar_music}] {int(score_music)} | "
                     f"POD: [{bar_podcast}] {int(score_podcast)}")
    sys.stdout.flush()

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- AUTO EQ HMM MONITOR ---")
    print("Carregando modelos...")
    
    try:
        hmm_music = joblib.load('models/hmm_music.pkl')
        hmm_podcast = joblib.load('models/hmm_podcast.pkl')
    except:
        print("\nERRO: Modelos nao encontrados na pasta 'models/'.")
        return

    print("Iniciando monitoramento de audio...")
    
    try:
        mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    except:
        mic = sc.all_microphones(include_loopback=True)[0]

    print(f"Ouvindo dispositivo: {mic.name}")
    print("Pressione CTRL+C para parar.\n")

    current_mode = "music" # Estado inicial assumido

    with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
        while True:
            try:
                data = recorder.record(numframes=int(SAMPLE_RATE * DURATION))
                
                if np.max(np.abs(data)) < 0.01:
                    sys.stdout.write(f"\r[ SILENCIO ] Aguardando audio...{' '*50}")
                    sys.stdout.flush()
                    continue

                features = extract_features_from_buffer(data, SAMPLE_RATE)

                score_music = hmm_music.score(features)
                score_podcast = hmm_podcast.score(features)

                new_mode = current_mode
                if score_music > score_podcast:
                    new_mode = 'music'
                else:
                    new_mode = 'podcast'

                changed = update_apo_config(new_mode)
                if changed:
                    current_mode = new_mode
                
                print_debug_status(score_music, score_podcast, current_mode)

            except KeyboardInterrupt:
                print("\n\nParando monitoramento...")
                break
            except Exception as e:
                print(f"\nErro no loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()