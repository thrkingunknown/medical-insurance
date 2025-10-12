# 🏥 MediPredict - Medical Insurance Cost Predictor

A production-ready, enterprise-grade full-stack application that predicts medical insurance costs using advanced machine learning techniques. Features comprehensive data preprocessing, ensemble modeling, and a modern React interface.

Built with FastAPI, React, TypeScript, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.3-green.svg)
![React](https://img.shields.io/badge/React-19.1.0-61dafb.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-ff69b4.svg)

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

### Core Capabilities

- 🤖 **Ensemble ML Model**: Advanced stacking ensemble combining Random Forest, Gradient Boosting, and LightGBM for superior accuracy
- 🧹 **Enterprise-Grade Data Preprocessing**: Removes 51.77% duplicate records, handles outliers intelligently, validates data quality
- 🔧 **Advanced Feature Engineering**: Creates 24+ engineered features (5× feature expansion) for richer pattern recognition
- � **Real-time Predictions**: Instant insurance cost estimates with 84%+ accuracy
- 📈 **Robust Scaling**: Uses RobustScaler for outlier-resistant normalization

### User Experience

- 🎨 **Modern UI**: Clean, responsive interface built with React 19 and Tailwind CSS
- 🚀 **Fast API**: High-performance backend with automatic Swagger documentation
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- 🔒 **Input Validation**: Client and server-side validation ensures data integrity
- 🌓 **Dark Mode**: Built-in theme toggle for user preference
- ⚡ **Lightning Fast**: Vite-powered frontend with hot module replacement

### Data Quality

- ✅ **Zero Missing Values**: Complete data validation pipeline
- ✅ **No Duplicates**: Intelligent duplicate detection and removal
- ✅ **Outlier Handling**: Statistical IQR method with smart capping
- ✅ **Data Consistency**: Standardized categorical values
- ✅ **Range Validation**: All inputs validated against realistic bounds

## 🛠️ Tech Stack

### Backend

- **FastAPI 0.118.3** - Modern, high-performance web framework for building APIs
- **scikit-learn 1.7.2** - Machine learning library for model training and evaluation
- **LightGBM 4.6.0** - Gradient boosting framework for high-performance ML
- **Pandas 2.3.3** - Data manipulation and analysis
- **NumPy 2.3.3** - Numerical computing and array operations
- **SciPy 1.11+** - Statistical analysis and outlier detection
- **Pydantic 2.12.0** - Data validation using Python type annotations
- **Uvicorn 0.37.0** - Lightning-fast ASGI server

### Frontend

- **React 19.1.0** - Modern UI library with latest features
- **TypeScript 5.9.3** - Type-safe JavaScript for robust code
- **Vite** - Next-generation frontend build tool with HMR
- **Tailwind CSS** - Utility-first CSS framework for rapid UI development
- **Radix UI** - Accessible, unstyled UI component primitives
- **Lucide React** - Beautiful, consistent icon library
- **Vercel Analytics** - Real-time web analytics

### Development Tools

- **Python 3.11+** - Modern Python with type hints and performance improvements
- **Node.js 18+** - JavaScript runtime for frontend tooling
- **ESLint** - Code linting and quality checks
- **PostCSS** - CSS transformation and optimization

## 📊 Data Preprocessing

The application includes enterprise-grade data preprocessing for optimal model performance:

### 1. Data Cleaning

- **Duplicate Removal**: Removes 1,435 duplicate records (51.77% of original dataset)
- **Categorical Standardization**: Converts all text to lowercase and trims whitespace
- **Data Validation**: Validates age (18-64), BMI (15-50), and children count (0-5)
- **Quality Assurance**: Ensures 100% data integrity with zero missing values

### 2. Outlier Handling

- **BMI Outliers**: Caps extreme values using realistic medical bounds (15-50)
- **Charge Outliers**: Applies IQR method (3×IQR) for statistical outlier detection
- **Z-Score Validation**: Identifies extreme values beyond 3 standard deviations
- **Smart Capping**: Preserves data points while ensuring realistic values

### 3. Feature Engineering (24 New Features)

**Age-Based Features (6):**

- `age_group` - Categorical bins (young, adult, middle, senior)
- `age²`, `age³` - Polynomial transformations for non-linear patterns
- `log_age` - Logarithmic transformation for skewed distributions
- `is_senior`, `is_young` - Binary age indicators

**BMI-Based Features (6):**

- `bmi_category` - WHO categories (underweight, normal, overweight, obese)
- `bmi²`, `bmi³` - Polynomial features for non-linear relationships
- `log_bmi` - Log transformation
- `is_obese`, `bmi_deviation` - Health indicators

**Interaction Features (5):**

- `age_bmi` - Combined age and BMI effects
- `age_smoker` - Age-smoking interaction
- `bmi_smoker` - BMI-smoking interaction
- `age_children`, `bmi_children` - Family size interactions

**Risk Indicators (5):**

- `smoker_obese` - High-risk combination
- `senior_smoker` - Elderly smoker risk
- `senior_obese` - Elderly obesity risk
- `young_smoker` - Young smoker indicator
- `risk_score` - Weighted composite risk metric

**Binary Flags (2):**

- `is_smoker` - Smoking status (0/1)
- `has_children` - Dependent status (0/1)

### 4. Scaling & Normalization

- **RobustScaler**: Outlier-resistant scaling using median and IQR
- **OneHotEncoder**: Categorical variable encoding with unknown value handling
- **Train-Test Split**: 80/20 split with random shuffling
- **Feature Count**: 6 original → 30 total features (5× expansion)

### Data Quality Metrics

| Metric          | Before         | After    | Improvement     |
| --------------- | -------------- | -------- | --------------- |
| Total Records   | 2,772          | 1,337    | 48.2% (unique)  |
| Duplicates      | 1,435 (51.77%) | 0 (0%)   | ✅ 100% cleaned |
| BMI Outliers    | 6 extreme      | 3 capped | ✅ 50% reduced  |
| Charge Outliers | 14 extreme     | 8 capped | ✅ 43% reduced  |
| Features        | 6              | 30       | ✅ 5× expansion |
| Missing Values  | 0              | 0        | ✅ Maintained   |

## 🤖 Machine Learning Models

The application uses an advanced ensemble learning approach with four sophisticated models:

### Model Architecture

1. **Random Forest Regressor**

   - Ensemble of 100 decision trees
   - Bootstrap sampling for variance reduction
   - Feature importance ranking
   - Robust to overfitting

2. **Gradient Boosting Regressor**

   - Sequential tree building
   - Adaptive learning rate (0.1)
   - 100 estimators with max depth 5
   - Strong predictive performance

3. **LightGBM (Light Gradient Boosting Machine)**

   - High-speed gradient boosting framework
   - Leaf-wise tree growth strategy
   - Memory-efficient histogram-based learning
   - Excellent handling of categorical features

4. **Stacking Regressor (Meta-Model)**
   - Combines predictions from all base models
   - Ridge regression as final estimator
   - Cross-validation (5-fold) for robustness
   - Leverages strengths of all models

### Training Pipeline

1. **Data Loading**: Reads `insurance.csv` (2,772 original records)
2. **Preprocessing**: Applies `clean_data()` → 1,337 unique records
3. **Feature Engineering**: Creates 24 new features via `preprocess_data()`
4. **Scaling**: RobustScaler for numeric features
5. **Encoding**: OneHotEncoder for categorical variables
6. **Training**: Fits all four models on processed data
7. **Serialization**: Saves models and preprocessors to disk

### Model Performance

- **R² Score**: ~0.85-0.90 (varies by model)
- **MAE**: Mean Absolute Error tracking
- **Cross-Validation**: 5-fold validation for reliability
- **Feature Importance**: Top features include smoking status, age, BMI

## 🧪 Testing

The project includes comprehensive automated testing to ensure data quality and preprocessing reliability.

### Test Suite (`test_preprocessing.py`)

**Test Coverage:**

1. **Import Verification**

   - ✅ All required libraries import successfully
   - ✅ Module dependencies validated

2. **Data Loading**

   - ✅ CSV file reads correctly (2,772 records)
   - ✅ All expected columns present
   - ✅ No missing values

3. **Data Cleaning (`clean_data()`)**

   - ✅ Removes duplicates: 2,772 → 1,337 (48.2% retention)
   - ✅ Caps BMI outliers (15-50 range)
   - ✅ Handles charge outliers (IQR method)
   - ✅ Standardizes categorical variables

4. **Feature Engineering (`preprocess_data()`)**
   - ✅ Creates all 24 engineered features
   - ✅ Generates polynomial features (age², age³, bmi², bmi³)
   - ✅ Builds interaction features (age_bmi, age_smoker, etc.)
   - ✅ Calculates risk indicators
   - ✅ Final feature count: 30 total

### Running Tests

```powershell
cd backend
python test_preprocessing.py
```

**Expected Output:**

```
Imports ✅
Data Loading ✅
Data Cleaning ✅
Feature Engineering ✅

All preprocessing tests passed! ✅
```

### Analysis Tools

**`data_preprocessing_report.py`** - Comprehensive before/after analysis:

- Generates detailed statistical comparison
- Shows duplicate reduction metrics
- Displays outlier detection results
- Exports preprocessing summary

**`visualize_preprocessing.py`** - Visual quality assessment:

- Creates distribution comparison charts
- Generates before/after plots
- Produces correlation heatmaps
- Saves visualizations to `preprocessing_results/`

**`analyze_data.py`** - Initial data quality check:

- Identifies duplicates and outliers
- Validates data ranges
- Suggests preprocessing steps

## 📁 Project Structure

```
medical-insurance/
├── backend/
│   ├── app.py                           # FastAPI main application
│   ├── insurance.csv                    # Training dataset (2,772 records)
│   ├── requirements.txt                 # Python dependencies
│   ├── test_preprocessing.py            # Automated test suite
│   ├── data_preprocessing_report.py     # Analysis reporting tool
│   ├── visualize_preprocessing.py       # Visualization generator
│   ├── analyze_data.py                  # Data quality checker
│   ├── models/                          # Serialized ML models
│   │   ├── random_forest_model.pkl
│   │   ├── gradient_boosting_model.pkl
│   │   ├── lightgbm_model.pkl
│   │   └── stacking_model.pkl
│   ├── preprocessors/                   # Fitted transformers
│   │   ├── scaler.pkl
│   │   └── encoder.pkl
│   └── __pycache__/                     # Python bytecode cache
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── InsuranceForm.tsx       # Main prediction form
│   │   │   ├── ThemeProvider.tsx       # Dark/light mode context
│   │   │   ├── ThemeToggle.tsx         # Theme switcher button
│   │   │   ├── GithubButton.tsx        # Repository link component
│   │   │   └── ui/                     # Reusable UI components
│   │   │       ├── button.tsx
│   │   │       ├── input.tsx
│   │   │       ├── label.tsx
│   │   │       └── select.tsx
│   │   ├── lib/
│   │   │   └── utils.ts                # Utility functions (cn helper)
│   │   ├── assets/                     # Static images/icons
│   │   ├── App.tsx                     # Root React component
│   │   ├── App.css                     # App-specific styles
│   │   ├── index.css                   # Global styles + Tailwind
│   │   ├── main.tsx                    # React entry point
│   │   └── vite-env.d.ts              # Vite type declarations
│   ├── public/                         # Static assets
│   ├── index.html                      # HTML template
│   ├── package.json                    # Node dependencies
│   ├── vite.config.ts                  # Vite configuration
│   ├── tsconfig.json                   # TypeScript config (base)
│   ├── tsconfig.app.json               # App TypeScript config
│   ├── tsconfig.node.json              # Node TypeScript config
│   ├── tailwind.config.js.backup       # Tailwind backup config
│   ├── postcss.config.js               # PostCSS configuration
│   └── eslint.config.js                # ESLint rules
│
└── README.md                            # Project documentation
```

### Key Files Explained

**Backend:**

- **`app.py`** - Main FastAPI application with ML prediction endpoints, preprocessing functions (`clean_data()`, `preprocess_data()`), and model training logic
- **`insurance.csv`** - Original training dataset with 2,772 records (age, sex, BMI, children, smoker, region, charges)
- **`test_preprocessing.py`** - Automated testing suite validating data cleaning, feature engineering, and preprocessing pipeline
- **`data_preprocessing_report.py`** - Generates comprehensive before/after analysis with statistical comparisons
- **`visualize_preprocessing.py`** - Creates visual charts comparing data distributions before/after preprocessing
- **`analyze_data.py`** - Initial data quality assessment tool for identifying duplicates and outliers
- **`models/*.pkl`** - Serialized trained models (Random Forest, Gradient Boosting, LightGBM, Stacking Regressor)
- **`preprocessors/*.pkl`** - Fitted transformers (RobustScaler for numeric features, OneHotEncoder for categorical)

**Frontend:**

- **`InsuranceForm.tsx`** - Interactive prediction form with real-time validation and API integration
- **`ThemeProvider.tsx`** - Context provider for persistent dark/light mode state
- **`ThemeToggle.tsx`** - Theme switcher button with smooth transitions
- **`GithubButton.tsx`** - Repository link component with icon
- **`ui/*`** - Shadcn/ui components with Tailwind styling and Radix UI accessibility
- **`utils.ts`** - Utility functions including className merger (`cn`)
- **`vite.config.ts`** - Vite build configuration with React plugin and optimizations

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
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# With specific Python installation
C:/Python313/python.exe -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
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

## 🧪 Development

### Backend Development

```bash
# Run with auto-reload
cd backend
python -m uvicorn app:app --reload

# Run preprocessing tests
python test_preprocessing.py

# Generate preprocessing analysis report
python data_preprocessing_report.py

# Create visualization charts
python visualize_preprocessing.py

# Analyze data quality
python analyze_data.py

# Format code (if black is installed)
black app.py

# Type checking (if mypy is installed)
mypy app.py
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

Edit `backend/app.py` to configure:

- **CORS Origins**: Modify `allow_origins` in `app.add_middleware()` for production domains
- **Model Parameters**: Adjust estimators, learning rates, or add new models in `train_models()`
- **Preprocessing Logic**: Customize `clean_data()` and `preprocess_data()` functions
- **Feature Engineering**: Add new features or modify existing ones in the preprocessing pipeline
- **API Settings**: Configure FastAPI metadata, versioning, and documentation URLs

### Frontend Configuration

Edit configuration files to customize frontend behavior:

- **`vite.config.ts`**: Proxy settings, build options, development server port
- **`tailwind.config.js.backup`**: Tailwind CSS theme customization (colors, fonts, breakpoints)
- **`tsconfig.json`**: TypeScript compiler options and module resolution
- **`postcss.config.js`**: PostCSS plugins and transformations

## 🐛 Troubleshooting

### Backend Issues

**Port already in use:**

```powershell
# Change the port number
python -m uvicorn app:app --reload --port 8001
```

**Module not found:**

```powershell
# Reinstall dependencies
cd backend
pip install -r requirements.txt
```

**Model training errors:**

```powershell
# Verify data file exists
ls insurance.csv

# Check data integrity
python analyze_data.py

# Manually retrain models
python app.py
```

**Preprocessing test failures:**

```powershell
# Run tests with verbose output
python test_preprocessing.py

# Check if scipy is installed (required for outlier detection)
pip show scipy

# Reinstall scipy if needed
pip install scipy>=1.11.0
```

### Frontend Issues

**Port 5173 in use:**

The frontend will automatically try the next available port (5174, 5175, etc.)

**Build errors:**

```powershell
# Clear cache and reinstall (Windows PowerShell)
cd frontend
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

**CORS errors:**

- Ensure backend is running on http://localhost:8000
- Check that CORS middleware is configured in `backend/app.py`
- Verify frontend is making requests to the correct API URL

**TypeScript errors:**

```powershell
# Rebuild TypeScript declarations
npm run build
```

## 🤝 Contributing

Contributions are welcome! Whether you want to add new features, improve preprocessing, or enhance the UI, we appreciate your help.

### How to Contribute

1. **Fork the Project** - Click the "Fork" button at the top right of the repository
2. **Clone Your Fork**
   ```powershell
   git clone https://github.com/your-username/medical-insurance.git
   cd medical-insurance
   ```
3. **Create a Feature Branch**
   ```powershell
   git checkout -b feature/AmazingFeature
   ```
4. **Make Your Changes**
   - Add new preprocessing features to `backend/app.py`
   - Update tests in `backend/test_preprocessing.py`
   - Enhance UI components in `frontend/src/components/`
   - Update documentation in `README.md`
5. **Test Your Changes**

   ```powershell
   # Backend tests
   cd backend
   python test_preprocessing.py

   # Frontend build
   cd frontend
   npm run build
   ```

6. **Commit Your Changes**
   ```powershell
   git add .
   git commit -m 'Add some AmazingFeature'
   ```
7. **Push to GitHub**
   ```powershell
   git push origin feature/AmazingFeature
   ```
8. **Open a Pull Request** - Go to your fork on GitHub and click "New Pull Request"

### Contribution Ideas

- 🧹 **Data Quality**: Improve outlier detection algorithms
- 🔧 **Feature Engineering**: Add domain-specific medical features
- 🤖 **ML Models**: Experiment with neural networks or XGBoost
- 🎨 **UI/UX**: Enhance form validation and error messages
- 📊 **Visualization**: Add charts showing prediction breakdowns
- 🧪 **Testing**: Increase test coverage and add integration tests
- 📚 **Documentation**: Improve code comments and API docs

## � License

This project is open source and available for educational and commercial use.

## �📧 Contact

For questions, feedback, or collaboration opportunities:

- **Issues**: Open an issue on GitHub for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for general questions and ideas
- **Pull Requests**: Submit PRs for code contributions

---

### Built With

- **Backend**: FastAPI 0.118.3 | scikit-learn 1.7.2 | LightGBM 4.6.0 | SciPy 1.11+
- **Frontend**: React 19 | TypeScript 5.9.3 | Vite | Tailwind CSS
- **ML Pipeline**: RobustScaler | OneHotEncoder | Ensemble Stacking
- **Data Quality**: 48.2% data cleaning | 24 engineered features | 5× feature expansion

Made with ❤️ for better healthcare cost predictions
