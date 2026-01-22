## 1. Model Generator (`my_torch_generator`)

This tool generates an initial `.npz` file containing the architecture blueprint and randomized weights.

### Usage
```bash
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
config_file: Path to the YAML configuration file.

nb: Number of network instances to generate based on this config.

Configuration Format (YAML)
The generator expects a YAML file defining the model structure:

```YAML
model:
  name: "deep_chess_v1"
  directory_path: "./models"
  initialization:
    method: "xavier"
    seed: 42
  architecture:
    input_size: 836
    output_size: 3
    layers:
      - size: 256
        activation: "LeakyReLU"
        dropout: 0.1
      - size: 128
        activation: "LeakyReLU"
    output:
      activation: "Softmax"
```


## 2. Chess Analyzer (`my_torch_analyzer`)
This tool is the core engine for training the model or predicting outcomes based on chess positions.

### Usage

### Training Mode
Loads a dataset, trains the model, and saves the updated weights.

```bash
./my_torch_analyzer --train --threads 4 --save models/trained.npz models/init.npz data/chess_data.fen
```

### Prediction Mode
Predicts outcomes for a file of positions.

```bash
./my_torch_analyzer --predict models/trained.npz data/test_positions.fen
```

### Debug Mode
Runs prediction with detailed analysis (Confusion Matrix, probabilities). Requires labeled data.


```bash
./my_torch_analyzer --predict --debug models/trained.npz data/labeled_test.fen
```

### Arguments
--train: Enable training mode.

--predict: Enable prediction mode.

--save FILE: (Optional) Path to save the trained model.

--threads N: Number of threads for parallel data loading and evaluation (CPU).

--label-smoothing FLOAT: Applies label smoothing (0.0 to 1.0) to reduce overfitting.

LOADFILE: Input .npz file containing the model (weights + blueprint).

CHESSFILE: Input file containing chess positions.

### Data Format (FEN)
The CHESSFILE must contain one position per line.

Format: FEN_STRING [RESULT]

Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Checkmate White

### Technical Features
Feature Engineering: The input vector is not just the raw board; it includes king safety metrics, piece mobility, and X-ray attack detection.

Autosufficient Models: The .npz files contain both the weights and the architecture blueprint, allowing the Analyzer to reconstruct the model without the original YAML config.

Parallelization: Uses ThreadPoolExecutor to speed up batch evaluation.