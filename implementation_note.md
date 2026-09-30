# Task 5 Implementation Note

## Objective
Build a deep-learning/NLP text classifier satisfying the RabTech Task 5 brief.

## Dataset
IMDB movie reviews with binary sentiment labels.

## Preprocessing
The TensorFlow IMDB loader provides integer-encoded reviews. The notebook decodes them into normalized text and then fits Keras `TextVectorization` only on the training split. Validation and test text are transformed using the learned vocabulary.

## Model
The classifier uses:
- Embedding
- GlobalAveragePooling1D
- Dense(128, ReLU)
- BatchNormalization
- Dropout(0.35)
- Dense(64, ReLU)
- BatchNormalization
- Dropout(0.30)
- Dense(1, sigmoid)

## Training
Binary cross-entropy is optimized with Adam. EarlyStopping restores the best validation-loss weights. ReduceLROnPlateau adapts the learning rate when validation loss plateaus.

## Evaluation
The notebook reports accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix on the held-out test set.

## Inference
Four custom review texts are passed to the trained pipeline. Each prediction includes the predicted sentiment, positive probability and confidence.

## Reproducibility
A fixed random seed is used. The learned vocabulary, training history, metrics, inference outputs, metadata and trained Keras model are saved under `artifacts/`.
