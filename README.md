<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/krislette/bach-or-bot">
    <img src="https://raw.githubusercontent.com/krislette/bob-web/refs/heads/main/public/bob.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Bach or Bot</h1>
  <p align="center">
    MusicLIME and Multimodal MLP Framework For Explainable Classification Of AI-Generated And Human-Composed Music
    <br />
    <a href="https://drive.google.com/file/d/132gWKb6OsOgqYAiq3chzpTqNo24Ob_AQ/view?usp=sharing"><strong>Explore the paper »</strong></a>
    <br />
    <br />
    <a href="https://bach-or-bot-tool.vercel.app/">View Demo</a>
    ·
    <a href="https://github.com/krislette/bach-or-bot/issues">Report Bug</a>
    ·
    <a href="https://github.com/krislette/bach-or-bot/pulls">Request Feature</a>
  </p>
</div>

## Overview

A multimodal machine learning system that classifies music as human-composed or AI-generated using both audio features and lyrics analysis. This project implements a comprehensive pipeline combining state-of-the-art audio processing, spectro-temporal audio analysis, natural language processing, and explainable AI techniques to provide accurate and interpretable music classification.

## Inspiration

Bach or Bot addresses the growing challenge of distinguishing between human-composed and AI-generated music in an era where artificial intelligence can create increasingly sophisticated musical content. The system employs a multimodal approach that analyzes both audio characteristics and lyrical content to make classification decisions, which provides explanations for its predictions through advanced explainability techniques.

### Key Features

- **Multimodal Classification**: Combines audio and lyrics analysis for improved accuracy
- **Audio-Only Classification**: Supports classification and explanation using only audio features
- **Explainable AI**: Implements MusicLIME for interpretable model predictions
- **RESTful API**: FastAPI-based server with comprehensive endpoints
- **Optimized Performance**: Combined endpoints with shared processing for efficiency
- **Scalable Architecture**: Modular design supporting different model configurations

## Architecture

The system follows a multimodal fusion architecture:

1. **Audio and Lyrics Preprocessing**: Raw audio files are resampled and normalized, while lyrics text is cleaned and tokenized for consistent input formatting
2. **Audio Feature Extraction**: SpecTTTra (Spectro-Temporal Tokens Transformer) extracts audio features from spectrograms
3. **Lyrics Feature Extraction**: LLM2Vec generates semantic embeddings from song lyrics
4. **Feature Fusion**: Intermediate fusion layer combines audio and text representations
5. **Classification**: Multi-layer perceptron (MLP) performs final binary classification
6. **Explainability**: MusicLIME provides feature-level explanations for predictions

### Technical Implementation

- **Audio Pipeline**: Raw audio → Preprocessing → SpecTTTra feature extraction → Scaling → Classification
- **Lyrics Pipeline**: Raw text → Preprocessing → LLM2Vec embeddings → PCA reduction → Scaling → Classification
- **Multimodal Fusion**: Concatenated audio and lyrics features → MLP classifier
- **Explainability**: Source separation via OpenUnmix → Perturbation analysis → LIME explanations

## Technologies and References

This project builds upon several state-of-the-art research contributions:

### Core Models

**SpecTTTra**: Spectro-Temporal Tokens Transformer for audio representation learning

- _Reference_: Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., & Fattah, S. A. (2024). "SONICS: Synthetic Or Not -- Identifying Counterfeit Songs." arXiv preprint arXiv:2408.14080. Accepted to ICLR 2025.
- _Link_: https://arxiv.org/abs/2408.14080
- _Repository_: https://github.com/awsaf49/sonics

**LLM2Vec**: Large Language Model to Vector conversion for text embeddings

- _Reference_: BehnamGhader, P., et al. (2024). "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders." arXiv preprint arXiv:2404.05961.
- _Link_: https://arxiv.org/abs/2404.05961
- _Repository_: https://github.com/McGill-NLP/llm2vec

**MusicLIME**: Local Interpretable Model-agnostic Explanations for music classification

- _Reference_: Sotirou, T., et al. (2024). "MusicLIME: Explainable Multimodal Music Understanding." arXiv preprint arXiv:2409.10496. To be presented at ICASSP 2025.
- _Link_: https://arxiv.org/abs/2409.10496
- _Repository_: https://github.com/IamTheo2000/MusicLIME

### Supporting Technologies

- **Multi-Layer Perceptron (MLP)**: Neural network architecture for classification
- **OpenUnmix**: Source separation for audio factorization in explainability
- **LIME**: Local Interpretable Model-agnostic Explanations

## Project Structure

```bash
bach-or-bot/
├── app/                            # FastAPI application
│ ├── schemas.py                    # Pydantic response models
│ ├── server.py                     # API endpoints and server configuration
│ ├── utils.py                      # Server utility functions
│ └── validators.py                 # Input validation functions
├── config/                         # Configuration files
│ ├── data_config.yml               # Data processing parameters
│ ├── model_config.yml              # Model hyperparameters
│ └── server_config.yml             # Server configuration
├── data/
│ ├── external/                     # External data and test samples
│ ├── processed/                    # Preprocessed datasets
│ └── raw/                          # Original datasets
├── docs/                           # Documentation
├── models/                         # Trained model artifacts
│ ├── fusion/                       # Fusion layer models and scalers
│ ├── mlp/                          # MLP classifier checkpoints
│ ├── musiclime/                    # MusicLIME model artifacts
│ └── spectttra/                    # SpecTTTra model checkpoints
├── notebooks/
│ ├── exploratory/                  # Data exploration and MusicLIME research
│ ├── inference/                    # Model inference notebooks
│ └── modeling/                     # Model development notebooks
├── scripts/                        # Execution scripts
│ ├── evaluate.py                   # Model evaluation
│ ├── explain.py                    # Explanation generation (multimodal, unimodal, combined)
│ ├── explain_runner.py             # Explanation testing script
│ ├── explain_combined_runner.py    # Combined explanation testing
│ ├── predict.py                    # Prediction pipeline (multimodal, unimodal, combined)
│ ├── predict_runner.py             # Prediction testing script
│ └── train.py                      # Training pipeline
├── src/                            # Source code modules
│ ├── llm2vectrain/                 # LLM2Vec training and inference
│ │ ├── model.py
│ │ ├── llm2vec_trainer.py
│ │ └── config.py
│ ├── models/                       # Model definitions
│ │ └── mlp.py                      # MLP classifier
│ ├── musiclime/                    # MusicLIME implementation
│ │ ├── explainer.py                # Core MusicLIME explainer
│ │ ├── factorization.py            # Audio source separation
│ │ ├── text_utils.py               # Text processing utilities
│ │ └── wrapper.py                  # Prediction wrappers
│ ├── preprocessing/                # Data preprocessing modules
│ │ ├── audio_preprocessor.py
│ │ ├── lyrics_preprocessor.py
│ │ └── preprocessor.py
│ ├── spectttra/                    # SpecTTTra implementation
│ │ ├── spectttra.py
│ │ ├── spectttra_trainer.py
│ │ ├── feature.py
│ │ ├── embedding.py
│ │ ├── tokenizer.py
│ │ └── transformer.py
│ └── utils/                        # Utility functions
├── tests/                          # Unit and integration tests
│ ├── test_preprocessing.py
│ ├── test_features.py
│ ├── test_mlp.py
│ ├── test_spectttra.py
│ └── test_musiclime.py
├── Dockerfile                      # Container configuration
├── Dockerfile.hf                   # Hugging Face deployment configuration
├── pyproject.toml                  # Project dependencies and metadata
└── README.md                       # This file!
```

## Installation and Setup

### Prerequisites

- Python 3.11 to 3.13 exclusively
- Poetry
- Git
- 8GB+ RAM recommended
- GPU recommended for faster inference (optional)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/krislette/bach-or-bot.git
   cd bach-or-bot
   ```

2. **Install dependencies using Poetry**:

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry env activate
   ```

## Configuration

### Environment Variables

- `MUSICLIME_NUM_SAMPLES`: Number of perturbation samples for LIME (default: 1000)
- `MUSICLIME_NUM_FEATURES`: Number of top features to return (default: 10)
- `HF_TOKEN`: Huggingface token for LLM2Vec access

### Model Configuration

Edit `config/model_config.yml` to adjust model hyperparameters:

- MLP architecture (hidden layers, dropout, activation)
- Training parameters (learning rate, batch size, epochs)

### Server Configuration

Edit `config/server_config.yml` to adjust server settings:

- File upload limits
- CORS settings
- API rate limiting

## Usage

### API Server

Start the FastAPI server:

```bash
poetry run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Available Endpoints

#### Prediction Endpoints

_Multimodal Prediction_

- `POST /api/v1/predict/multimodal` - Prediction using both audio and lyrics
- `POST /api/v1/predict` - Legacy endpoint (same as multimodal)

_Audio-Only Prediction_

- `POST /api/v1/predict/audio` - Prediction using only audio features

_Combined Prediction_

- `POST /api/v1/predict/combined` - Both predictions in one call (convenience wrapper)

#### Explanation Endpoints

_Multimodal Explanation_

- `POST /api/v1/explain/multimodal` - MusicLIME explanation using both modalities
- `POST /api/v1/explain` - Legacy endpoint (same as multimodal)

_Audio-Only Explanation_

- `POST /api/v1/explain/audio` - MusicLIME explanation using only audio

_Combined Explanation (Optimized_

- `POST /api/v1/explain/combined` - Both explanations with shared source separation (~50% faster)

#### Information Endpoints

- `GET /` - API welcome message and endpoint listing
- `GET /api/v1/model/info` - Model information and capabilities

### API Usage Examples

#### Multimodal Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/multimodal" \
 -H "Content-Type: multipart/form-data" \
 -F "audio_file=@path/to/song.mp3" \
 -F "lyrics=Your song lyrics here"
```

**Response Format:**

```json
{
  "status": "success",
  "lyrics": "Your song lyrics here",
  "audio_file_name": "song.mp3",
  "audio_content_type": "audio/mpeg",
  "audio_file_size": 1234567,
  "results": {
    "confidence": 0.8542,
    "prediction": "Human-Composed",
    "label": 1,
    "probability": 0.8542
  }
}
```

#### Audio-Only Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/audio" \
 -H "Content-Type: multimodal/form-data" \
 -F "audio_file=@path/to/song.mp3"
```

#### Multimodal Explanation

```bash
curl -X POST "http://localhost:8000/api/v1/explain/multimodal" \
 -H "Content-Type: multipart/form-data" \
 -F "audio_file=@path/to/song.mp3" \
 -F "lyrics=Your song lyrics here"
```

**Response Format:**

```json
{
  "status": "success",
  "lyrics": "Your song lyrics here",
  "audio_file_name": "song.mp3",
  "results": {
    "prediction": {
      "class": 1,
      "class_name": "Human-Composed",
      "confidence": 0.8542,
      "probabilities": [0.1458, 0.8542]
    },
    "explanations": [
      {
        "rank": 1,
        "modality": "audio",
        "feature_text": "vocals",
        "weight": 0.2341,
        "importance": 0.2341
      },
      {
        "rank": 2,
        "modality": "lyrics",
        "feature_text": "Line 1: Your song lyrics here",
        "weight": 0.1876,
        "importance": 0.1876
      }
    ],
    "summary": {
      "total_features_analyzed": 10,
      "audio_features_count": 6,
      "lyrics_features_count": 4,
      "runtime_seconds": 45.23,
      "samples_generated": 1000,
      "timestamp": "2024-10-26T09:15:30"
    }
  }
}
```

#### Combined Explanation (Optimized)

```bash
curl -X POST "http://localhost:8000/api/v1/explain/combined" \
 -H "Content-Type: multipart/form-data" \
 -F "audio_file=@path/to/song.mp3" \
 -F "lyrics=Your song lyrics here"
```

**Response Format:**

```json
{
   "status": "success",
   "results": {
      "multimodal": {
         "prediction": { /_ multimodal prediction results / },
         "explanations": [ / multimodal explanations / ],
         "summary": { / multimodal processing summary / }
      },
      "audioonly": {
         "prediction": { /_ audio-only prediction results / },
         "explanations": [ / audio-only explanations / ],
         "summary": { / audio-only processing summary / }
      },
      "combinedsummary": {
         "total_runtime_seconds": 67.45,
         "factorization_time_seconds": 42.1,
         "source_separation_reused": true,
         "timestamp": "2024-10-26T09:15:30"
      }
   }
}
```

### Command Line Usage

#### Running Predictions

Test multimodal and audio-only predictions:

```bash
poetry run python -m scripts.predict_runner
```

#### Running Explanations

Test explanation generation:

```bash
poetry run python -m scripts.explain_runner
```

#### Test combined predictions

```bash
poetry run python -m scripts.predict_combined_runner
```

#### Test combined explanations (optimized with shared source separation)

```bash
poetry run python -m scripts.explain_combined_runner
```

#### Model Training

Train the complete pipeline:

```bash
poetry run python -m scripts.train.py
```

### Deployment

Build and run using Docker:

```bash
# Build the container
docker build -t bach-or-bot .

# Run the container
docker run -p 8000:8000 bach-or-bot
```

### Hugging Face Spaces

The project includes a Hugging Face deployment configuration:

```bash
docker build -f Dockerfile.hf -t bach-or-bot-hf .
```

## Model Performance

The system provides classification confidence scores and detailed explanations for each prediction. The multimodal approach typically achieves higher accuracy than audio-only classification by leveraging complementary information from both modalities.

### Performance Optimizations

- **Combined Explanations**: The `/explain/combined` endpoint performs source separation once and reuses it for both multimodal and audio-only explanations which reduces processing time by approximately 50%
- **Batch Processing**: MusicLIME uses optimized batch processing for perturbation analysis
- **Model Caching**: Predictors load models once and reuse them across multiple requests

### Explainability Features

- **Feature Importance**: Identifies which audio components and lyric lines contribute most to predictions
- **Source Separation**: Uses OpenUnmix to isolate different audio sources (vocals, drums, bass, other)
- **Temporal Analysis**: Analyzes audio features across different time segments
- **Lyric Analysis**: Highlights important lyrical phrases and semantic content
- **Modality Comparison**: Compare predictions and explanations between multimodal and audio-only approaches

## Development

### Adding Dependencies

Add runtime dependencies:

```bash
poetry add package-name
```

Add development dependencies:

```bash
poetry add --group dev package-name
```

### Testing

Run the test suite:

```bash
poetry run pytest tests/
```

Run specific test modules:

```bash
poetry run pytest tests/test_musiclime.py
poetry run pytest tests/test_spectttra.py
poetry run pytest tests/test_mlp.py
```

### Code Quality

The project follows Python best practices:

- PEP 8 style guidelines
- Comprehensive docstrings following NumPy/SciPy format
- Type hints where applicable
- Modular architecture for maintainability
- Separation of concerns between API, models, and utilities

## Contributors

This project was developed by a collaborative team of researchers and developers:

- **Acelle Krislette Rosales** (acellekrislette@gmail.com) - MusicLIME implementation, explainability features, API and containerization, and final optimizations
- **Hans Christian Queja** (hansqueja8@gmail.com) - Preprocessing, SpecTTTra integration, training, and final optimizations
- **Regina Bonifacio** (bonifacioregina06@gmail.com) - MLP classifier architecture and training, and pytest testing
- **Sean Matthew Sinalubong** (s3amatth3wsinalubong@gmail.com) - LLM2Vec implementation and training
- **Syruz Ken Domingo** (syruzkenc.domingo@gmail.com) - SpecTTTra implementation and optimization

## Citation

If you use this work in your research, please cite:

```bibtex
@software{bach_or_bot_2025,
   title = {Bach or Bot: MusicLIME and Multimodal MLP Framework For Explainable Classification Of AI-Generated And Human-Composed Music},
   author = {Rosales, Acelle Krislette and Queja, Hans Christian and Bonifacio, Regina and Sinalubong, Sean Matthew and Domingo, Syruz Ken},
   year = {2025},
   url = {https://github.com/krislette/bach-or-bot}
}
```

## Acknowledgments

We acknowledge the research contributions that made this project possible, particularly the authors of [SpecTTTra](https://arxiv.org/abs/2408.14080), [LLM2Vec](https://arxiv.org/abs/2404.05961), and [MusicLIME](https://arxiv.org/abs/2409.10496). This work builds upon their foundational research in audio processing, natural language processing, and explainable AI for music understanding.

Special thanks to the open-source community for providing the tools and libraries that enabled this research, including OpenUnmix for source separation, LIME for explainability frameworks, and the broader PyTorch ecosystem for deep learning infrastructure.

## Contact

For questions or collaboration inquiries, please contact the development team through the repository issues page.

## License

Distributed under the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) License. See [LICENSE](LICENSE) for more information.

<p align="right">[<a href="#readme-top">Back to top</a>]</p>
