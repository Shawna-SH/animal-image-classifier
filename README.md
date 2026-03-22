# animal-image-classifier

End-to-end cat-vs-dog image classification project with model training, evaluation, single-image prediction, and a Streamlit web app.

## Overview

This project uses a ResNet18 image classifier built with PyTorch and the Oxford-IIIT Pet dataset.

Main entrypoints:

- `python -m src.train` for model training
- `python -m src.evaluate` for evaluation and reports
- `python -m src.predict` for single-image inference
- `streamlit run app/app.py` for the web app

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

The dataset is loaded through `torchvision.datasets.OxfordIIITPet`.

- Training will automatically download the dataset into `data/raw/` if needed.
- The default classification task is binary: `cat` vs `dog`.

## Train the model

Run training from the project root:

```bash
python -m src.train
```

What it does:

- builds a ResNet18 classifier
- trains on the training split
- evaluates on the validation split during training
- saves the best checkpoint to `artifacts/models/best_model.pth`
- prints final test metrics at the end

Current default training settings in code:

- epochs: `5`
- batch size: `32`
- learning rate: `1e-3`

## Evaluate a trained model

Run evaluation on the test split:

```bash
python -m src.evaluate \
  --checkpoint artifacts/models/best_model.pth \
  --split test
```

Run evaluation on the validation split:

```bash
python -m src.evaluate \
  --checkpoint artifacts/models/best_model.pth \
  --split val
```

Useful optional arguments:

```bash
python -m src.evaluate \
  --checkpoint artifacts/models/best_model.pth \
  --output-dir artifacts/evaluation \
  --data-dir data/raw \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 0 \
  --device auto \
  --split test
```

Evaluation outputs are saved under `artifacts/evaluation/`:

- `{split}_report.txt`
- `{split}_confusion_matrix.png`
- `{split}_y_true.npy`
- `{split}_y_pred.npy`
- `{split}_y_prob.npy`

## Predict a single image

Run prediction on one image:

```bash
python -m src.predict data/raw/oxford-iiit-pet/images/Egyptian_Mau_167.jpg
```

Example with explicit checkpoint:

```bash
python -m src.predict \
  data/raw/oxford-iiit-pet/images/pug_52.jpg \
  --checkpoint artifacts/models/best_model.pth
```

The CLI prints:

- input image path
- predicted label
- confidence score
- per-class probabilities

## Run the Streamlit app

Start the web app from the project root:

```bash
streamlit run app/app.py
```

The app lets you:

- upload one image (`jpg`, `jpeg`, `png`)
- preview the uploaded image
- classify it as `cat` or `dog`
- view confidence and top probabilities
