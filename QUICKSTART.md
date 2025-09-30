# AIRL Franka Cube Lift Project

## 🚀 Schnellstart

### 1. Dependencies installieren
```bash
cd /home/chris/IsaacLab_mik/Projects/airl_franka
pip install -r requirements.txt
```

### 2. Expert-Daten vorbereiten
```bash
# Expert-Daten validieren oder dummy-Daten erstellen
python utils/validate_expert_data.py --file data/expert_data.hdf5 --create-dummy
```

### 3. Training starten
```bash
# Mit Standard-Konfiguration
python train_airl.py

# Mit angepasster Konfiguration
python train_airl.py --config config.yaml

# Training fortsetzen von Checkpoint
python train_airl.py --resume logs/checkpoints/latest_checkpoint.pt
```

### 4. Modell evaluieren
```bash
# Policy evaluieren
python utils/evaluate_policy.py --checkpoint logs/checkpoints/latest_checkpoint.pt --episodes 20 --plot

# Mit Visualisierung (falls Isaac Lab verfügbar)
python utils/evaluate_policy.py --checkpoint logs/checkpoints/latest_checkpoint.pt --render
```

## 📁 Projektstruktur

```
airl_franka/
├── agents/
│   ├── discriminator.py      # AIRL Discriminator
│   └── policy.py             # PPO Policy & Value Function
├── envs/
│   └── franka_wrapper.py     # Isaac Lab Environment Wrapper
├── utils/
│   ├── gae.py               # Generalized Advantage Estimation
│   ├── replay_buffer.py     # Experience Replay Buffer
│   ├── validate_expert_data.py  # Expert Data Validation
│   └── evaluate_policy.py   # Policy Evaluation
├── data/                    # Expert Demonstration Data
├── logs/
│   ├── checkpoints/         # Model Checkpoints
│   └── tensorboard/         # Training Logs
├── config.yaml             # Training Configuration
├── train_airl.py          # Main Training Script
├── setup.py              # Package Installation
└── requirements.txt      # Python Dependencies
```

## ⚙️ Konfiguration

Die Hauptkonfiguration befindet sich in `config.yaml`:

- **Environment**: Franka Cube Lift Task Einstellungen
- **Network**: Policy und Discriminator Architektur
- **Training**: PPO und AIRL Hyperparameter
- **Logging**: Tensorboard und Checkpoint Einstellungen

## 🎯 Expert-Daten Format

Expert-Daten müssen im HDF5-Format vorliegen mit folgenden Datasets:
- `obs`: Observations (N x obs_dim)
- `acts`: Actions (N x act_dim)  
- `next_obs`: Next Observations (N x obs_dim)
- `episode_starts`: Episode Boundaries (optional)

## 📊 Monitoring

### Tensorboard
```bash
tensorboard --logdir logs/tensorboard
```

### Training Logs
- Console Output: Episoden-Metriken
- File Logs: `logs/tensorboard/training.log`
- Checkpoints: `logs/checkpoints/`

## 🔧 Development

### Package installieren (entwicklungsmodus)
```bash
pip install -e .
```

### Tests ausführen
```bash
# Expert-Daten validieren
python utils/validate_expert_data.py --file data/expert_data.hdf5

# Environment testen
python -c "from envs.franka_wrapper import make_env; env=make_env(); print('Environment loaded successfully')"

# Dummy Training (1 Episode)
python train_airl.py --config config.yaml  # Stoppe nach 1 Episode mit Ctrl+C
```

## 🐛 Troubleshooting

### Isaac Lab nicht verfügbar
Das Projekt funktioniert auch ohne Isaac Lab mit einer dummy-Environment für Entwicklung/Testing.

### CUDA-Probleme
```bash
# CPU-only Training erzwingen
python train_airl.py --device cpu
```

### Memory Issues
Reduziere in `config.yaml`:
- `env.num_envs` (Anzahl parallele Umgebungen)
- `training.ppo.batch_size` (Batch-Größe)
- `training.ppo.buffer_size` (Buffer-Größe)

## 📝 Nächste Schritte

1. **Expert-Daten sammeln**: Trainiere einen PPO Agent oder sammle manuelle Demonstrationen
2. **Isaac Lab Integration**: Passe `envs/franka_wrapper.py` an echte Isaac Lab Environment an
3. **Hyperparameter tuning**: Experimentiere mit verschiedenen Einstellungen in `config.yaml`
4. **Evaluation**: Nutze `utils/evaluate_policy.py` für detaillierte Performance-Analyse

## 🔗 Nützliche Links

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [AIRL Paper](https://arxiv.org/abs/1710.11248)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
