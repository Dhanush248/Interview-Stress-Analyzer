# Real-Time AI-Based Interview Stress and Confidence Analysis System

A complete end-to-end system that conducts real-time interviews with AI-powered stress and confidence analysis using WebRTC, multimodal machine learning, and modern web technologies.

## 🎯 Project Overview

This system enables real-time video interviews between interviewers and interviewees while providing AI-based behavioral analysis exclusively to interviewers. The system analyzes facial expressions and voice patterns to determine stress levels and confidence scores in real-time.

### Key Features

- **Real-time WebRTC Video Calls**: High-quality peer-to-peer video communication
- **AI-Powered Analysis**: Multimodal stress and confidence detection
- **Privacy-First Design**: Analysis results visible only to interviewers
- **Live Analytics Dashboard**: Real-time charts and recommendations
- **Temporal Smoothing**: Stable predictions using rolling averages
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │◄──►│  Node.js Server │◄──►│ Python AI Server│
│   (Frontend)    │    │   (WebRTC +     │    │   (FastAPI +    │
│                 │    │   WebSocket)    │    │   ML Models)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Frontend (React)**: User interface with WebRTC integration
2. **Node.js Backend**: WebRTC signaling and real-time communication
3. **Python AI Backend**: Machine learning inference engine
4. **ML Models**: Face CNN + Voice LSTM + Fusion Model

## 🧠 AI/ML Architecture

### Datasets Used

- **FER-2013**: Facial expression recognition (48x48 grayscale images)
- **RAVDESS**: Audio emotion recognition (WAV files)

### Model Pipeline

1. **Face Model**: Lightweight CNN for facial stress detection
2. **Voice Model**: Bidirectional LSTM with attention for voice analysis
3. **Fusion Model**: Combines face and voice features for final prediction

### Stress Mapping

- **Low Stress**: Happy, Neutral, Calm emotions
- **Medium Stress**: Surprise, Disgust emotions
- **High Stress**: Angry, Fear, Sad emotions

## 🚀 Quick Start

### Prerequisites

- **Python 3.10.x** (REQUIRED - other versions not supported)
- **Node.js 16+** and npm
- **Git**
- **Webcam and microphone**

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Interview Stress Analyser"
```

### 2. Setup Python Environment

```bash
# Run the automated setup script
setup_python.bat

# Or manually:
python -m venv ai_env
ai_env\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Datasets (Optional - for training)

If you want to train models from scratch:

1. **FER-2013**: Download from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Place `fer2013.csv` in `datasets/fer2013/`

2. **RAVDESS**: Download from [Kaggle RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
   - Extract to `datasets/ravdess/audio_speech_actors_01-24/`

### 4. Train Models (Optional)

```bash
# Activate Python environment
ai_env\Scripts\activate

# Train all models
cd backend\ai_server
python train_models.py

# Normal training (skips preprocessing if data exists)
python train_models.py

# Force reprocessing of datasets
python train_models.py --force_preprocessing

# Only preprocess, don't train
python train_models.py --preprocessing_only

# Skip preprocessing entirely (use existing data)
python train_models.py --skip_preprocessing

# Quick run - skips everything already done
python train_models.py

# Only retrain fusion model (if it failed)
python train_models.py --force_training

# Force retrain everything (if needed)
python train_models.py --force_training --force_preprocessing


# Or train with custom paths
python train_models.py --fer2013_csv "path/to/fer2013.csv" --ravdess_dir "path/to/ravdess"
```

### 5. Install Node.js Dependencies

```bash
cd backend\node_server
npm install

cd ..\..\frontend
npm install
```

### 6. Start the System

```bash
# Option 1: Start everything at once
start_full_system.bat

# Option 2: Start services individually
start_ai_backend.bat      # Terminal 1
start_node_server.bat     # Terminal 2
cd frontend && npm start  # Terminal 3
```

### 7. Access the Application

- **Frontend**: http://localhost:3001
- **Node.js Server**: http://localhost:3000
- **AI Backend**: http://localhost:8001

## 📋 Usage Instructions

### For Interviewers

1. Go to http://localhost:3001
2. Enter your name and select "Interviewer"
3. Generate or enter a room ID
4. Share the room ID with the interviewee
5. Start the video call
6. View real-time stress analytics in the right panel

### For Interviewees

1. Go to http://localhost:3001
2. Enter your name and select "Interviewee"
3. Enter the room ID provided by interviewer
4. Join the interview room
5. Participate normally (AI analysis is hidden)

## 🔧 Configuration

### AI Model Settings

Edit `backend/ai_server/inference_engine.py`:

```python
# Temporal smoothing window
self.stress_history = deque(maxlen=5)  # Last 5 predictions

# Analysis frequency
frameIntervalRef.current = setInterval(() => {
    captureAndSendFrame();
}, 2000); // Every 2 seconds
```

### WebRTC Configuration

Edit `frontend/src/components/VideoCall.js`:

```javascript
const rtcConfig = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    // Add TURN servers for production
  ],
};
```

## 📊 Model Performance

### Face Model (CNN)

- **Architecture**: 4 Conv layers + BatchNorm + Dropout
- **Input**: 48x48 grayscale images
- **Classes**: 3 stress levels
- **Expected Accuracy**: ~65-75%

### Voice Model (LSTM)

- **Architecture**: Bidirectional LSTM + Attention
- **Features**: MFCC, Pitch, Energy, Spectral features
- **Classes**: 3 stress levels
- **Expected Accuracy**: ~60-70%

### Fusion Model

- **Architecture**: Feature concatenation + FC layers
- **Input**: Face (64-dim) + Voice (32-dim) features
- **Output**: Stress level + Confidence score
- **Expected Accuracy**: ~70-80%

## 🛠️ Development

### Project Structure

```
Interview Stress Analyser/
├── backend/
│   ├── ai_server/          # Python FastAPI backend
│   │   ├── models/         # ML model definitions
│   │   ├── preprocessing/  # Dataset preprocessing
│   │   ├── main.py        # FastAPI server
│   │   └── inference_engine.py
│   └── node_server/        # Node.js WebRTC server
│       ├── server.js      # Express + Socket.IO
│       └── package.json
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── App.js
│   └── package.json
├── models/trained/         # Trained model files
├── datasets/              # Training datasets
├── scripts/               # Automation scripts
└── docs/                  # Documentation
```

### Adding New Features

1. **New ML Model**: Add to `backend/ai_server/models/`
2. **New Analysis**: Extend `inference_engine.py`
3. **Frontend Changes**: Modify React components
4. **API Changes**: Update FastAPI endpoints

## 🔍 Troubleshooting

### Common Issues

**1. Python Version Error**

```
ERROR: Python 3.10.x is required
```

**Solution**: Install Python 3.10.x from python.org

**2. Model Files Not Found**

```
FileNotFoundError: Face model not found
```

**Solution**: Train models first or download pre-trained models

**3. WebRTC Connection Failed**

```
ICE connection failed
```

**Solution**: Check firewall settings, add TURN servers for production

**4. AI Backend Connection Error**

```
AI backend connection closed
```

**Solution**: Ensure Python environment is activated and models are loaded

### Performance Issues

**1. High CPU Usage**

- Reduce analysis frequency in VideoCall.js
- Use smaller video resolution
- Optimize model inference

**2. Memory Issues**

- Clear analysis history regularly
- Reduce temporal smoothing window
- Monitor browser memory usage

## 📝 API Documentation

### Node.js Server Endpoints

- `GET /`: Health check
- `GET /health`: Detailed status
- `GET /api/rooms`: List active rooms
- `POST /api/rooms/create`: Create new room
- `GET /api/ai/test`: Test AI backend connection

### Python AI Server Endpoints

- `GET /`: Health check
- `GET /health`: Detailed health status
- `POST /analyze/image`: Analyze single image
- `POST /analyze/reset`: Reset analyzer history
- `WebSocket /ws/{client_id}`: Real-time analysis

### WebSocket Events

- `join-room`: Join interview room
- `offer/answer/ice-candidate`: WebRTC signaling
- `video-frame`: Send video frame for analysis
- `audio-chunk`: Send audio data
- `stress_analysis`: Receive analysis results

## 🎓 Academic Information

### Research Applications

- Human-Computer Interaction studies
- Behavioral analysis research
- Interview process optimization
- Stress detection methodologies

### Citation

```bibtex
@misc{interview_stress_analyzer_2024,
  title={Real-Time AI-Based Interview Stress and Confidence Analysis System},
  author={Your Name},
  year={2024},
  note={Final Year Project}
}
```

### Ethical Considerations

- Informed consent required
- Data privacy compliance
- Bias mitigation in AI models
- Transparent analysis methodology

## 📄 License

This project is for academic purposes. Please ensure compliance with dataset licenses:

- FER-2013: Academic use only
- RAVDESS: Academic research license

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

For issues and questions:

1. Check troubleshooting section
2. Review logs in browser console
3. Check Python/Node.js terminal outputs
4. Create GitHub issue with detailed description

---

**Built with ❤️ for academic research and real-world applications**
