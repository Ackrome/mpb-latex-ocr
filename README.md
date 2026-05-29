# MPB LaTeX OCR

Training scaffold for formula-image to LaTeX recognition.

The first implemented baseline is intentionally small: a CNN image encoder plus a Transformer decoder trained with PyTorch Lightning and tracked with MLflow. It is meant to validate data, metrics, and experiment hygiene before moving to larger UniMERNet-style or Hugging Face encoder-decoder models.

For a paper-style architecture description and diagram, see [docs/model_architecture.md](docs/model_architecture.md).

See [wiki.md](wiki.md) for setup, training, evaluation, prediction, MLflow, and hardware-profile usage.
