# 🚦 Lagos Traffic Analysis System

An AI-powered real-time vehicle detection and classification system designed specifically for Lagos, Nigeria roads. Detects and classifies Lagos-specific vehicle types including Okada, Danfo, BRT, Keke Napep, and more.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Real-time Vehicle Detection** - Live video stream processing with YOLOv8
- **Lagos-Specific Classification** - Identifies local vehicle types:
  - 🏍️ Okada (Motorcycles)
  - 🚐 Danfo (Yellow Minibuses)
  - 🚌 BRT (Blue Rapid Transit Buses)
  - 🛺 Keke Napep (Yellow Tricycles)
  - 🚍 Other Buses
  - 🚚 Trucks
  - 🚗 Private Cars
- **Unique Vehicle Counting** - IoU-based tracking counts each vehicle once
- **Professional Web Dashboard** - Modern UI with Tailwind CSS
- **Real-time Analytics** - Live charts and statistics
- **Database Logging** - SQLite storage for historical analysis

## 🖼️ Screenshots

*Add your screenshots here*

## 🛠️ Tech Stack

- **Detection**: YOLOv8 (Ultralytics)
- **Backend**: FastAPI + Uvicorn
- **Video Processing**: OpenCV
- **Database**: SQLite
- **Frontend**: HTML5 + Tailwind CSS + Chart.js
- **Real-time Updates**: WebSocket

## 📁 Project Structure

```
lagos_traffic/
├── config.py              # Configuration settings
├── main.py                # Main orchestration script
├── requirements.txt       # Python dependencies
├── modules/
│   ├── detector.py        # Vehicle detection & classification
│   ├── tracker.py         # IoU-based vehicle tracking
│   ├── camera.py          # Video/camera handling
│   └── database.py        # SQLite database operations
├── dashboard/
│   ├── app.py             # FastAPI backend
│   ├── templates/
│   │   └── index.html     # Web dashboard UI
│   └── static/            # Static assets
├── test_videos/           # Place test videos here
├── database/              # SQLite database storage
└── logs/                  # Application logs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lagos-traffic-analysis.git
   cd lagos-traffic-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add test videos**
   - Place `.mp4` or `.avi` files in the `test_videos/` folder
   - Videos should contain Lagos traffic footage for best results

### Running the Dashboard

```bash
cd dashboard
python app.py
```

Open your browser and navigate to: **http://localhost:8081**

### Running Standalone Detection

```bash
python main.py
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "yolov8n.pt"        # YOLOv8 model variant
CONFIDENCE_THRESHOLD = 0.35      # Detection confidence

# Performance settings
PROCESSING_WIDTH = 640           # Frame width for detection
STREAM_WIDTH = 960               # Frame width for streaming
SKIP_FRAMES = 5                  # Process every Nth frame
STREAM_FPS = 10                  # Stream frame rate

# Color detection (HSV)
YELLOW_HSV_LOWER = (20, 100, 100)   # Danfo/Keke detection
BLUE_HSV_LOWER = (100, 50, 50)      # BRT detection
```

## 🧠 How It Works

### Vehicle Classification Logic

1. **YOLOv8 Detection** - Detects base vehicle classes (car, motorcycle, bus, truck)

2. **Lagos-Specific Classification**:
   - **Okada**: Motorcycle detected
   - **Danfo**: Bus + Yellow color dominant
   - **BRT**: Bus + Blue color + Long aspect ratio
   - **Keke Napep**: Motorcycle/Car + Yellow + Black stripes + Small size
   - **Truck**: Truck detected
   - **Private Car**: Car detected (not matching other criteria)

3. **IoU Tracking** - Tracks vehicles across frames to count each unique vehicle once

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
- [Tailwind CSS](https://tailwindcss.com/)

## 📧 Contact

Your Name - [@Akeem Jr Odebiyi](https://linkedin.com/akeem-jr-odebiti)

Project Link: [https://github.com/Techdee1/Lagos_Traffic_AI_System](https://github.com/Techdee1/Lagos_Traffic_AI_System)

---

**Made with ❤️ in Nigeria 🇳🇬**
