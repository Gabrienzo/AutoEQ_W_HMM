import time
import sys
import os
import warnings
import numpy as np
import librosa
import joblib
import pyaudiowpatch as pyaudio

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES ---
APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"
SAMPLE_RATE = 22050 # Taxa interna para o modelo HMM
DURATION = 2.0      # Duração do buffer

def update_apo_config(mode, preset_music_path, preset_podcast_path):
    try:
        preset_path = preset_music_path if mode == 'music' else preset_podcast_path
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
        return False

def print_debug_status(score_music, score_podcast, active_mode):
    diff = abs(score_music - score_podcast)
    max_len = 30
    
    if score_music > score_podcast:
        confidence = min(diff / 500, 1.0)
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
    print("--- AUTO EQ HMM MONITOR (PyAudioWPatch) ---")
    
    # 1. SETUP DE CAMINHOS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_music_model = os.path.join(script_dir, 'models', 'hmm_music.pkl')
    path_podcast_model = os.path.join(script_dir, 'models', 'hmm_podcast.pkl')
    preset_music = os.path.join(script_dir, 'presets', 'music_preset.txt')
    preset_podcast = os.path.join(script_dir, 'presets', 'podcast_preset.txt')
    
    # 2. CARREGAR MODELOS
    print("Carregando modelos HMM...")
    try:
        hmm_music = joblib.load(path_music_model)
        hmm_podcast = joblib.load(path_podcast_model)
    except Exception as e:
        print(f"\nERRO CRÍTICO: Não foi possível ler os modelos em {script_dir}\\models")
        return

    # 3. SETUP DO PYAUDIO (LOOPBACK)
    print("Configurando WASAPI Loopback...")
    p = pyaudio.PyAudio()
    
    try:
        # Pega informações do sistema de áudio WASAPI
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        
        # Pega o dispositivo de SAÍDA padrão (Alto-falantes)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers["isLoopbackDevice"]:
            # Se o padrão não for loopback, procuramos o "irmão gêmeo" dele que seja loopback
            found = False
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    found = True
                    break
            
            if not found:
                print("ERRO: Não achei o dispositivo Loopback correspondente ao alto-falante padrão.")
                print("Tentando pegar o primeiro Loopback disponível...")
                # Fallback: pega o primeiro loopback que achar
                for loopback in p.get_loopback_device_info_generator():
                    default_speakers = loopback
                    break

        print(f"Dispositivo: {default_speakers['name']}")
        
        native_rate = int(default_speakers["defaultSampleRate"])
        native_channels = default_speakers["maxInputChannels"]
        print(f"Nativo: {native_rate}Hz / {native_channels} Canais")
        
        # Calcula tamanho do buffer
        CHUNK_SIZE = int(native_rate * DURATION)

        # Abre o stream
        stream = p.open(format=pyaudio.paFloat32,
                        channels=native_channels,
                        rate=native_rate,
                        input=True,
                        input_device_index=default_speakers["index"],
                        frames_per_buffer=CHUNK_SIZE)
        
        print("\nIniciando Monitoramento... (CTRL+C para parar)")
        current_mode = "music"

        while True:
            # Lê os bytes do PyAudio
            # exception_on_overflow=False evita crash se o PC ficar lento por 1ms
            raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Converte bytes para Numpy Array
            data = np.frombuffer(raw_data, dtype=np.float32)
            
            # O PyAudio retorna um array flat (tripa). Precisamos remodelar se for estéreo
            # Mas para MFCC precisamos de mono e flat de qualquer jeito.
            
            # Se for estéreo/multicanal, os dados vêm intercalados [L, R, L, R...]
            # Para transformar em mono rápido, pegamos a média
            if native_channels > 1:
                # Remodela para (frames, canais)
                data = data.reshape(-1, native_channels)
                # Tira a média dos canais (vira Mono)
                y_mono = np.mean(data, axis=1)
            else:
                y_mono = data

            # Checagem de silêncio
            if np.max(np.abs(y_mono)) < 0.01:
                sys.stdout.write(f"\r[ SILENCIO ] Aguardando audio...{' '*40}")
                sys.stdout.flush()
                continue

            # RESAMPLE (Se a placa for 48k/44k e o modelo 22k)
            if native_rate != SAMPLE_RATE:
                y_resampled = librosa.resample(y_mono, orig_sr=native_rate, target_sr=SAMPLE_RATE)
            else:
                y_resampled = y_mono

            # MFCC & SCORE
            mfcc = librosa.feature.mfcc(y=y_resampled, sr=SAMPLE_RATE, n_mfcc=13)
            features = mfcc.T

            score_music = hmm_music.score(features)
            score_podcast = hmm_podcast.score(features)

            if score_music > score_podcast:
                new_mode = 'music'
            else:
                new_mode = 'podcast'

            changed = update_apo_config(new_mode, preset_music, preset_podcast)
            if changed:
                current_mode = new_mode
            
            print_debug_status(score_music, score_podcast, current_mode)

    except KeyboardInterrupt:
        print("\nParando...")
    except Exception as e:
        print(f"\n[ERRO FATAL] {e}")
    finally:
        # Limpeza
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()