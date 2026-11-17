# Machine Condition Intelligence

![Machine Condition Intelligence](docs/model.png)

Machine Condition Intelligence is a compact model that identifies bearing faults from vibration. It reads a short window of accelerometer signal and returns the condition of the bearing. The model is built to stay accurate when the motor load, the noise level or the sensor position changes, adapts to a new sensor without labelled data, and flags input that no longer resembles its training data.

| | |
|---|---|
| **Input** | 2048 samples from one accelerometer at 12 kHz, about 0.17 seconds |
| **Output** | Normal, inner-race fault, ball fault or outer-race fault, with a confidence |
| **Format** | ONNX, 8-bit (`model-int8.onnx`, 41 KB) or 32-bit (`model.onnx`, 88 KB) |
| **Parameters** | 21,604 |
| **Runtime** | ONNX Runtime and NumPy, 0.02 ms per window on one CPU thread |

## How It Works

![Model architecture](docs/architecture.png)

**Input.** A window of 2048 acceleration samples from a sensor on the motor housing.

**Normalisation.** The window is shifted to zero mean and scaled to unit variance, so the model responds to the shape of the signal rather than its level. This step is part of the model, so the file needs no separate preprocessing.

**Frequency filters.** A wide first convolution, with a kernel of 64 samples and a stride of 8, learns 16 frequency filters from the raw signal. Bearing faults produce repeated impacts that excite specific frequency bands, and these filters learn which bands carry that information.

**Convolution.** Three blocks of convolution, batch normalisation, ReLU and max pooling combine the filtered signals into 64 channels that respond to fault patterns over time.

**Pooling.** Each channel is averaged over the window, giving 64 values that describe which patterns are present, wherever they occur.

**Classification.** A linear layer maps the 64 values to a score for each condition. The highest score is the prediction.

## Performance

Each change is tested on data the model has not seen, and scored with macro-F1 across the four conditions. A baseline of ten hand-crafted vibration features with gradient-boosted trees is shown for reference.

![Macro-F1 under a new motor load and under noise](reports/figures/comparison.png)

![Macro-F1 on the fan-end sensor, unseen in training, before and after adaptation](reports/figures/sensor-shift.png)

The 8-bit file makes the same decision as the 32-bit model on 100% of drive-end windows and 98.8% of fan-end windows.

## Adaptation and Monitoring

**Adaptation.** At a new sensor position, the model re-estimates its batch-normalisation statistics from unlabelled windows recorded there (AdaBN). No weights are retrained. This raises macro-F1 on the unseen sensor from 0.69 to 0.84.

**Drift detection.** Each batch of 32 windows is compared with the training data using the 64 pooled values. An alarm is raised when the difference exceeds the level that 99% of training batches stay below. The model raised no alarms for a new load or for noise at 0 dB, and flagged 75% of batches from the unseen sensor.

## Training

**Data.** Sixteen recordings from the [Case Western Reserve University bearing dataset](https://engineering.case.edu/bearingdatacenter): four conditions at four motor loads (0 to 3 hp), each with a drive-end and a fan-end accelerometer. The faults are 0.007 inch across and located in the drive-end bearing. Each recording is trimmed to 10 seconds and cut into windows, and every split holds out whole recordings so no recording is used for both training and testing.

**Augmentation.** In each epoch, about half of the training windows receive white noise at a random signal-to-noise ratio between 20 dB and −5 dB, and a random filter that simulates the path vibration takes to a sensor mounted elsewhere. Noise augmentation is what keeps the model accurate at −5 dB. The filter is what lets it transfer to a new sensor.

**Settings.** 15 epochs with AdamW, a learning rate of 0.001 and a batch size of 64. The released model is trained on all drive-end recordings.

## Usage

### Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib scikit-learn torch onnx onnxruntime pytest
```

### Classifying a Window

```python
from pathlib import Path
from src.deployment.inference import Classifier

classifier = Classifier(Path("models"))
condition, confidence = classifier(window)  # 2048 samples at 12 kHz
```

### Reproducing the Model

```bash
python -m src.acquisition.fetch          # download the recordings
python -m src.run convolution-transfer   # evaluate under each condition
python -m src.deployment.release         # train, export and verify the model
python -m pytest -q                      # run the tests
```

## Limitations

- Trained on one bearing type, one test rig and one fault size.
- Strong coloured noise reduces inner-race recall, since training noise is white.
- At a new sensor position, predictions are reliable only after adaptation.

## License

The code is released under the MIT License. The data belongs to the Case Western Reserve University Bearing Data Center and is not redistributed.
