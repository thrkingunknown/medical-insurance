# 🏥 MediPredict

# Medical Insurance Cost Predictor

A modern full-stack application that predicts medical insurance costs using machine learning. Built with FastAPI, React, TypeScript, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.3-green.svg)
![React](https://img.shields.io/badge/React-19.1.0-61dafb.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-ff69b4.svg)

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)

## ✨ Features

- 🤖 **Ensemble ML Model**: Advanced stacking ensemble combining Random Forest, Gradient Boosting, and LightGBM
- 📊 **Real-time Predictions**: Instant insurance cost estimates with high accuracy
- 🎨 **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- 🚀 **Fast API**: High-performance backend with automatic API documentation
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔒 **Input Validation**: Client and server-side validation for data integrity

## 🛠️ Tech Stack

### Backend

- **FastAPI** - Modern, fast web framework for building APIs
- **scikit-learn** - Machine learning library for model training
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Uvicorn** - ASGI server for running FastAPI

### Frontend

- **React 19** - UI library for building interactive interfaces
- **TypeScript** - Type-safe JavaScript
- **Vite** - Next-generation frontend tooling
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives
- **Lucide React** - Beautiful icon library

## 📁 Project Structure

```
medical-insurance/
├── backend/
│   ├── api.py                 # FastAPI application
│   ├── insurance.csv          # Training dataset
│   ├── requirements.txt       # Python dependencies
│   └── BACKEND_SETUP.md      # Backend documentation
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── lib/             # Utility functions
│   │   ├── App.tsx          # Main application
│   │   └── main.tsx         # Entry point
│   ├── package.json         # Node dependencies
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **npm or yarn** - Comes with Node.js

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd medical-insurance
```

### 2. Backend Setup

Navigate to the backend directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
# or
yarn install
```

## ▶️ Running the Application

### Quick Start (Both Services)

You can run both the backend and frontend simultaneously using two terminal windows:

#### Terminal 1 - Backend

```bash
cd backend

# Generic command (if python is in PATH)
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# OR specific Python installation
C:/Python313/python.exe -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Terminal 2 - Frontend

```bash
cd frontend
npm run dev
# or
yarn dev
```

The frontend will be available at:

- **App**: http://localhost:5173

### Individual Service Commands

#### Backend Only

```bash
cd backend

# Development mode with auto-reload
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# With specific Python installation
C:/Python313/python.exe -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Only

```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📚 API Documentation

Once the backend is running, you can access:

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### `GET /`

Health check endpoint

**Response:**

```json
{
  "message": "Medical Insurance Cost Predictor API is running!",
  "version": "1.0.0",
  "status": "healthy"
}
```

#### `POST /predict`

Predict insurance cost based on input parameters

**Request Body:**

```json
{
  "age": 30,
  "sex": "male",
  "bmi": 25.5,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```

**Response:**

```json
{
  "ensemble": 5345.67,
  "input_data": { ... },
  "modelAccuracies": {
    "rf": 77.01,
    "gb": 80.77,
    "lgb": 72.12,
    "ensemble": 84.18
  }
}
```

## 🤖 Machine Learning Models

The application uses an ensemble stacking approach for maximum prediction accuracy:

### Ensemble Components

1. **Random Forest** - Ensemble of decision trees with bootstrap aggregating
2. **Gradient Boosting** - Sequential ensemble method with gradient descent optimization
3. **LightGBM** - Gradient boosting framework using tree-based learning algorithms

### Ensemble Model

The ensemble model uses **Stacking Regressor** which:

- Trains Random Forest, Gradient Boosting, and LightGBM as base models
- Uses Ridge Regression as the meta-learner to combine predictions
- Provides superior accuracy by leveraging strengths of all three algorithms

### Features Used

- **Age**: Customer's age (0-120)
- **Sex**: Male or Female
- **BMI**: Body Mass Index (10.0-50.0)
- **Children**: Number of dependents (0-10)
- **Smoker**: Yes or No
- **Region**: Northeast, Northwest, Southeast, Southwest

### Model Training

Models are automatically trained on startup using the `insurance.csv` dataset. The training process includes:

- Data preprocessing with StandardScaler and OneHotEncoder
- 80/20 train-test split
- Model evaluation and comparison

## 🧪 Development

### Backend Development

```bash
# Run with auto-reload
cd backend
python -m uvicorn api:app --reload

# Run tests (if available)
pytest

# Format code
black api.py

# Type checking
mypy api.py
```

### Frontend Development

```bash
cd frontend

# Start dev server
npm run dev

# Lint code
npm run lint

# Type check
npx tsc --noEmit

# Build for production
npm run build
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/api.py` to configure:

- CORS origins
- Model parameters
- API settings

### Frontend Configuration

Edit `frontend/vite.config.ts` to configure:

- Proxy settings
- Build options
- Development server settings

## 🐛 Troubleshooting

### Backend Issues

**Port already in use:**

```bash
# Change the port number
python -m uvicorn api:app --reload --port 8001
```

**Module not found:**

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

**Port 5173 in use:**
The frontend will automatically try the next available port (5174, 5175, etc.)

**Build errors:**

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using FastAPI and React**
