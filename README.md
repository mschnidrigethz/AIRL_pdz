# PPO-AIRL für Franka Cube Stacking in Isaac Lab

## Projektübersicht

Dieses Projekt implementiert eine PPO-AIRL Pipeline für einen Cube Stacking Task mit einem Franka Arm in Isaac Lab.

- Die Umgebung ist als Wrapper im Ordner `envs/` hinterlegt.
- Policy und Discriminator sind im Ordner `agents/`.
- Expertendaten liegen im HDF5-Format im Ordner `data/`.
- Training und Hilfsfunktionen befinden sich in `train_airl.py` und `utils.py`.

## Ordnerstruktur

airl_franka/
├── agents/
│ ├── discriminator.py
│ └── policy.py
├── envs/
│ └── franka_wrapper.py
├── data/
│ └── expert_data.hdf5
├── utils.py
├── train_airl.py
└── README.md

## Installation

1. Python 3.8+ empfohlen  
2. Virtuelle Umgebung anlegen (optional):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Abhängigkeiten installieren:

pip install torch gym h5py numpy




Experten-Daten

Lege deine Expertendaten im HDF5-Format in data/expert_data.hdf5 ab. Die Datei sollte zwei Datasets enthalten: obs und acts.

Training starten
python train_airl.py

Das Skript trainiert PPO mit AIRL Discriminator Updates und gibt Episoden-Rewards aus.

Anpassungen

Passe envs/franka_wrapper.py an deine Isaac Lab Umgebung an.

Experten-Daten-Loader erwartet die Gruppen obs und acts in der HDF5-Datei.

Hyperparameter findest du in train_airl.py (Lernraten, Batchgrößen, etc.).
