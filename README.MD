```
# 🔍 NewsLens: Fake News Detector

A comprehensive fake news detection system powered by DistilBERT, featuring a web interface and Chrome extension for real-time news analysis.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Dataset Information](#dataset-information)
- [API Endpoints](#api-endpoints)
- [Chrome Extension](#chrome-extension)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

## ✨ Features

- **🤖 AI-Powered Detection**: Uses a fine-tuned DistilBERT model for accurate fake news classification.
- **🌐 Web Interface**: Interactive Gradio-based app for testing news articles.
- **🔌 Chrome Extension**: Real-time analysis of web content with browser integration.
- **⚡ Caching System**: SQLite-based caching for improved performance.
- **📊 Confidence Scoring**: Provides credibility scores and detailed predictions.
- **🔗 Source Verification**: Optional NewsAPI integration for source credibility checks.

## 🎯 Demo

### Web Interface
Access the Gradio interface at your Hugging Face Space URL after deployment (e.g., `https://yourusername-fake-news-detector.hf.space`).

### Chrome Extension
- Select text on any webpage.
- Right-click and choose "Analyze for Fake News".
- View results in the extension popup.

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Chrome browser (for extension)
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Download the trained model**
   - 📥 **Model Files**: [Download from Kaggle](https://www.kaggle.com/yourmodelpath) (replace with actual link).
   - 📂 Extract and place the following in `backend/models/`:
     - `config.json`
     - `model.safetensors`
     - `special_tokens_map.json`
     - `tokenizer_config.json`
     - `vocab.txt`

4. **Run the backend locally**
   ```bash
   python backend/app.py
   ```

### Chrome Extension Setup

1. Open Chrome Extensions:
   - Navigate to `chrome://extensions/`.
   - Enable "Developer mode".

2. Load the extension:
   - Click "Load unpacked".
   - Select the `extension/` folder from this project.

3. Start using:
   - The extension icon will appear in your toolbar.
   - Select text on any webpage and right-click to analyze.

## 📖 Usage

### Web Interface
1. Start the backend server:
   ```bash
   python backend/app.py
   ```
2. Open your browser and go to `http://127.0.0.1:7860` (local) or your Hugging Face Space URL.
3. Enter news text and optionally a source domain.
4. Click "Submit" to get credibility analysis.

### Chrome Extension
1. Navigate to any news website.
2. Select the text to analyze.
3. Right-click and choose "Analyze for Fake News".
4. View results in the popup.

### API Usage
Send POST requests to your hosted endpoint (e.g., `https://yourusername-fake-news-detector.hf.space/predict`) with:
```json
{
  "data": ["Your news text here", "optional-source-domain.com"]
}
```

## 📁 Project Structure
```
fake-news-detector/
├── backend/
│   ├── app.py                    # Gradio web interface
│   ├── requirements.txt          # Python dependencies
│   └── models/                   # Trained model files
│       ├── config.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── extension/                    # Chrome extension
│   ├── manifest.json            # Extension configuration
│   ├── background.js            # Background script
│   ├── content.js               # Content script
│   ├── popup.html               # Extension popup UI
│   └── popup.js                 # Popup functionality
├── training/                     # Model training scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Model evaluation
│   ├── visualize.py             # Results visualization
│   └── data/                    # Training datasets
│       ├── Fake.csv
│       ├── True.csv
│       ├── test.csv
│       └── train.csv
├── /data/cache.db               # SQLite cache database (persistent on Hugging Face)
└── README.md                    # Project documentation
```

## 🧠 Model Information

- **Architecture**: DistilBERT (Distilled BERT)
- **Task**: Binary sequence classification (Real vs Fake)
- **Training Data**: Custom dataset with fake and real news articles
- **Performance**: Optimized for speed and accuracy
- **Model Size**: ~268MB
- **Training Details**: Fine-tuned on DistilBERT base model with binary classification, evaluated using Accuracy, Precision, Recall, and F1-score.
- **Download model files(folder)**: - https://huggingface.co/spaces/Nikshey/FN-Detector/tree/main/backend/models

## 📊 Dataset Information

- **Source**: Custom dataset
- **Content**: 
  - Fake News Articles: Collected from unreliable sources
  - Real News Articles: Verified from credible sources
- **Format**: CSV files with text content and labels
- **Preprocessing**: Text cleaning and tokenization for DistilBERT
- **Download**: [Dataset Link](https://huggingface.co/spaces/Nikshey/FN-Detector/tree/main/training/data)

## 🔌 API Endpoints

- **Gradio Interface**
  - URL: Your Hugging Face Space URL (e.g., `https://yourusername-fake-news-detector.hf.space`)
  - Method: Web interface for interactive testing

- **API Prediction**
  - URL: `https://huggingface.co/spaces/Nikshey/FN-Detector`
  - Method: POST
  - Payload:
    ```json
    {
      "data": ["news text", "source domain (optional)"]
    }
    ```

## 🔧 Chrome Extension

- **Features**:
  - Text Selection Analysis
  - Context Menu Integration
  - Real-time Results
  - Popup Interface

- **Permissions**:
  - `activeTab`: Access current tab content
  - `scripting`: Inject content scripts
  - `storage`: Store analysis results
  - `contextMenus`: Add right-click menu options

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit changes: `git commit -m 'Add amazing feature'`.
4. Push to branch: `git push origin feature/amazing-feature`.
5. Open a Pull Request.

### Development Setup
- Follow the installation instructions.
- Make changes and test both web and extension components.
- Ensure all tests pass before submitting a PR.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for the DistilBERT implementation.
- Gradio for the web interface framework.
- NewsAPI for source credibility verification.
- Kaggle for hosting model files and dataset.
- DistilBERT Authors for the efficient architecture.
- Open Source Community for tools and libraries.

## 📞 Support
- Check the [Issues](https://github.com/NiksheyYadav/fake-news-detector/issues) page.
- Create a new issue with error messages and reproduction steps.
- ⭐ Star this repo if you find it helpful!
- [Email Me](mailto:relativity1905e@gmail.com)


## 📅 Version
- v1.0 (Last updated: August 01, 2025)
```
