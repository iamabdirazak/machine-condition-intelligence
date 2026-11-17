# Machine Condition Intelligence

Machine Condition Intelligence is a condition-monitoring project that investigates how machine-learning models can diagnose bearing faults from vibration signals while remaining reliable across changes in operating load, measurement noise and sensor position.

![Macro-F1 on the fan-end sensor, unseen in training, before and after adaptation](reports/figures/sensor-shift.png)

## Overview

Predictive maintenance depends on the ability to identify developing mechanical faults before they lead to equipment failure. Vibration-based machine learning offers a way to automate this process, but models trained under controlled conditions can become unreliable when deployed in a different operating environment.

A change in motor load, measurement noise or sensor position can alter the characteristics of the incoming signal. The model must distinguish these changes from the mechanical faults it is designed to recognise.

This project investigates that problem using bearing vibration measurements from the Case Western Reserve University (CWRU) Bearing Dataset.

A lightweight one-dimensional convolutional neural network processes 2,048-sample vibration windows. Its performance is evaluated across unseen operating loads, different noise levels and an unseen sensor position. The project also investigates unsupervised adaptation, input-drift detection and efficient CPU deployment.

## Results

The evaluation focuses on generalisation across operating conditions rather than relying exclusively on conventional random train/test splits.

Macro-F1 across the four conditions. Load results average four folds, and sensor results average four folds and three seeds.

![Macro-F1 under load shift and noise for each approach](reports/figures/comparison.png)

- **Load shift** is close to solved on this dataset. The feature baseline already reaches 1.00 when trained on three loads.
- **Noise** degrades the feature baseline to 0.40 at −5 dB. The network trained with noise augmentation holds 0.99 or above.
- **Sensor position** is the main challenge. Transfer augmentation and adaptation together raise macro-F1 on the unseen sensor from 0.31 (baseline) to 0.84.
- **Drift detection** raised no alarms for a new load or 0 dB noise, and flagged 75% of batches from the unseen sensor.

## Data

Sixteen recordings from the [Case Western Reserve University bearing dataset](https://engineering.case.edu/bearingdatacenter): four conditions at four motor loads (0 to 3 hp), with drive-end and fan-end accelerometers sampled at 12 kHz. Faults are 0.007 inch and located in the drive-end bearing. Only these files are downloaded and each is verified by checksum.

Each recording is trimmed to 10 seconds and cut into 2048-sample windows. Splits always hold out whole recordings, so no recording contributes to both training and testing.

## Method

The model is organised into independent stages for data acquisition, signal preparation, feature extraction, model development, training, evaluation, deployment and monitoring.

![Model architecture](docs/architecture.png)

**Baseline.** Ten hand-crafted features, six signal statistics and four envelope-spectrum fault-band energies, classified by gradient-boosted trees.

**Network.** A 1D convolutional network with 21,604 parameters. Normalisation is part of the model, so the exported file needs no preprocessing. A wide first layer learns frequency filters from the raw signal, three convolution blocks learn fault patterns, and pooling and a linear layer produce the class scores.

**Training.** About half of the training windows are augmented each epoch with white noise between 20 dB and −5 dB, and with a random filter that simulates a different transmission path to the sensor.

**Adaptation.** Batch-normalisation statistics are re-estimated from unlabelled windows at the new sensor (AdaBN). No weights are retrained.

**Drift detection.** Each batch of 32 windows is compared with the training data in the network's embedding space, with the alarm threshold set at the 99th percentile of training batches.

## Deployment

The released model is trained on all drive-end data, exported to ONNX and quantised to 8-bit integers, with the normalisation layer kept in full precision.

| File | Size | Latency per window | Agreement with PyTorch |
|---|---|---|---|
| `model.onnx` | 88 KB | 0.07 ms | 100% |
| `model-int8.onnx` | 41 KB | 0.04 ms | 100% drive end, 99.6% fan end |

Latency is measured on a single CPU thread. The model card is in `models/2026-11-17-network/model.json`.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib scikit-learn torch onnx onnxruntime pytest

python -m src.acquisition.fetch          # download the recordings
python -m src.run convolution-transfer   # evaluate an approach
python -m src.deployment.release         # train, export and verify the model
python -m pytest -q                      # run the tests
```

Inference requires only NumPy and ONNX Runtime.

```python
from pathlib import Path
from src.deployment.inference import Classifier

classifier = Classifier(Path("models/2026-11-17-network"))
condition, confidence = classifier(window)  # 2048 samples at 12 kHz
```

## Repository

```
data/       recordings, downloaded on demand
docs/       architecture diagram
models/     released model and model card
reports/    evaluation results and figures
src/        acquisition, preparation, features, modeling, training,
            evaluation, monitoring, deployment, visualization
tests/      split integrity and export parity
```

## Limitations

- One bearing type on one test rig, and one fault size.
- Noise augmentation uses white noise only. Strong coloured noise reduces inner-race recall.
- Adaptation requires unlabelled data from the new sensor before predictions are reliable.

## License

Code is released under the MIT License. The data belongs to the Case Western Reserve University Bearing Data Center and is not redistributed.
