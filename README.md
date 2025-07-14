<h1 align="center">Bach or Bot</h1>

A machine learning project that classifies music as human-composed or AI-generated using audio features and lyrics analysis.

## Project Structure

```
music-classifier/
├── .github/                    # GitHub workflows
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── config/                     # Configuration files
│   ├── model_config.yaml       # Model hyperparameters
│   └── data_config.yaml        # Data processing parameters
├── data/
│   ├── raw/                    # Original, unprocessed datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # External data sources
├── docs/                       # Documentation
│   ├── api.md                  # API documentation
│   └── model_architecture.md   # Model architecture details
├── logs/                       # Training and inference logs
├── models/                     # Trained model artifacts
│   ├── specttra/               # SpecTTTra model checkpoints
│   ├── llm2vec/                # LLM2Vec model checkpoints
│   └── fusion/                 # Fusion model checkpoints
├── notebooks/
│   ├── exploratory/            # Data exploration and visualization
│   ├── modeling/               # Model development and training
│   └── inference/              # Model deployment and predictions
├── results/                    # Experiment results and reports
├── scripts/                    # Execution scripts
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Model evaluation
│   └── predict.py              # Inference script
├── src/
│   ├── __init__.py
│   ├── preprocessing/          # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── audio_processor.py  # Audio preprocessing
│   │   └── lyrics_processor.py # Lyrics preprocessing
│   ├── features/               # Feature extraction modules
│   │   ├── __init__.py
│   │   ├── specttra.py         # SpecTTTra feature extraction
│   │   └── llm2vec.py          # LLM2Vec feature extraction
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── fusion.py           # Intermediate fusion layer
│   │   └── mlp.py              # MLP classifier
│   ├── explainability/         # Model interpretability
│   │   ├── __init__.py
│   │   └── music_lime.py       # MusicLIME implementation
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── common.py           # Common helper functions
│   └── api/                    # API endpoints
│       ├── __init__.py
│       └── endpoints.py        # FastAPI endpoints
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Container configuration
├── poetry.lock                 # Locked dependency versions
├── pyproject.toml              # Poetry configuration and dependencies
└── README.md                   # Project documentation
```

## Folder Structure Explanation

### Core Directories

- **`data/`**: Contains all datasets used in the project
  - `raw/`: Original, unprocessed audio files and lyrics
  - `processed/`: Cleaned and preprocessed data ready for training
  - `external/`: External datasets or reference data

- **`src/`**: Main source code directory
  - `preprocessing/`: Data cleaning and preprocessing modules
  - `features/`: Feature extraction implementations (SpecTTTra, LLM2Vec)
  - `models/`: Model architectures and training logic
  - `explainability/`: Model interpretability tools (MusicLIME)
  - `utils/`: Common utility functions and helpers
  - `api/`: API endpoints and server logic

- **`models/`**: Trained model artifacts and checkpoints
- **`config/`**: Configuration files for models and data processing
- **`scripts/`**: Standalone execution scripts for training and inference
- **`notebooks/`**: Jupyter notebooks for exploration and experimentation

### Supporting Directories

- **`tests/`**: Unit tests and integration tests
- **`docs/`**: Project documentation and API docs
- **`logs/`**: Training logs and system outputs
- **`results/`**: Experiment results, metrics, and visualizations

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry (install from https://python-poetry.org/docs/#installation)
- Git

### Fork and Clone Repository

1. **Fork the repository** on GitHub by clicking the "Fork" button

2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/music-classifier.git
   cd music-classifier
   ```

3. **Add upstream remote** to keep your fork updated:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/music-classifier.git
   ```

### Environment Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Activate virtual environment**:
   ```bash
   poetry env activate
   ```

   Or run commands directly:
   ```bash
   poetry run python scripts/train.py
   ```

### Data Setup

1. Place raw audio files in `data/raw/audio/`
2. Place lyrics files in `data/raw/lyrics/`
3. Run preprocessing scripts:
   ```bash
   python scripts/preprocess_data.py
   ```

## Contributing

### Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new functionality [optional]
- Update documentation when necessary

### Commit Message Format

- **Add/ed**: New features or files
- **Fix/ed**: Bug fixes
- **Update/d**: Changes to existing functionality
- **Remove/d**: Deleted code or files
- **Refactor/ed**: Code restructuring without functionality changes

### Testing

Run tests before submitting:
```bash
poetry run pytest tests/
```

### Adding Dependencies

Add new dependencies:
```bash
poetry add package-name
```

Add development dependencies:
```bash
poetry add --group dev package-name
```

## Team Responsibilities

- **Dataset Preprocessing**: Hans (audio, lyrics)
- **Feature Extraction**: 
  - SpecTTTra: Syke, Hans
  - LLM2Vec: Sean
- **Intermediate Fusion**: TBD (communicate assignment)
- **MLP Classifier**: Regina
- **MusicLIME**: Acelle
- **Model Packaging/API**: TBD
- **Integration**: TBD

## Usage

### Training

```bash
poetry run python scripts/train.py --config config/model_config.yaml
```

### Evaluation

```bash
poetry run python scripts/evaluate.py --model models/fusion/best_model.pth
```

### API Server

```bash
poetry run uvicorn src.api.endpoints:app --reload
```

## License

Soon (but I configured this to be MIT on poetry metadata)