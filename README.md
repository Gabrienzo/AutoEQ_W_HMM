# 🎧 AutoEQ HMM - Equalização Dinâmica Inteligente

> Um Agente Inteligente que monitora o áudio do Windows e adapta a equalização automaticamente entre **Música** e **Podcast** usando Inteligência Artificial.

## 📄 Sobre o Projeto

O **AutoEQ HMM** é um agente de software que roda em background no Windows. Ele captura o áudio do sistema em tempo real (Loopback), analisa as características sonoras usando **Modelos Ocultos de Markov (HMM)** e altera automaticamente o perfil do **Equalizer APO**.

O objetivo é proporcionar a melhor experiência auditiva sem intervenção manual: graves realçados para músicas e foco nos médios (voz) para podcasts/vídeos.

## 🚀 Funcionalidades

- **Monitoramento em Tempo Real:** Captura áudio digital via driver WASAPI (sem atrasos perceptíveis).
- **Classificação via IA:** Utiliza MFCCs e HMM para distinguir Música de Voz a cada 2 segundos.
- **Troca Automática:** Atualiza a configuração do Equalizer APO instantaneamente.
- **Robustez:** Funciona com fones cabeados, Bluetooth e caixas de som (utilizando `pyaudiowpatch`).
- **Eficiência:** Baixo consumo de CPU e proteção contra escritas desnecessárias em disco.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.12+
- **Captura de Áudio:** `pyaudiowpatch` (Suporte a WASAPI Loopback)
- **Processamento:** `librosa`, `numpy`
- **Machine Learning:** `hmmlearn` (Hidden Markov Models), `joblib`
- **Atuador:** Equalizer APO (Software de equalização para Windows)

## 📦 Estrutura do Projeto

```text
/AutoEQ_W_HMM
│
├── auto_eq.py              # Script principal do Agente (Código Fonte)
│
├── models/                 # Modelos treinados da IA
│   ├── hmm_music.pkl
│   └── hmm_podcast.pkl
│
└── presets/                # Arquivos de configuração de EQ
    ├── music_preset.txt    # Curva Harman / Bass Boost
    └── podcast_preset.txt  # Foco em Voz / Mid Range
