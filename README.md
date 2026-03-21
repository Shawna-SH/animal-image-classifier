# animal-image-classifier
End-to-end animal image classification project with model training, evaluation, and an interactive Streamlit demo.

## Setup
Create a virtual environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```
Install dependencies:
```
pip install -r requirements.txt
```

## Training
Train the model using:
```
python -m src.train
```
The trained model will be saved to:
```
artifacts/models/best_model.pth
```

## Evaluation
Evaluate the trained model on the test set:
```
python -m src.evaluate \
  --checkpoint artifacts/models/best_model.pth \
  --split test
```
Outputs will be saved to:
```
artifacts/evaluation/
```
Including:
- classification report
- confusion matrix
