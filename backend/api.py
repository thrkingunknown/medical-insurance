from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')

trained_models = {'lr': None, 'rf': None, 'gb': None, 'xgb': None, 'lgb': None, 'ensemble': None}
model_metrics = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting up FastAPI server...')
    train_models()
    yield
    print('Shutting down FastAPI server...')

app = FastAPI(title='MediPredict API - Enhanced', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:5174'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class InsuranceRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., pattern='^(male|female)$')
    bmi: float = Field(..., ge=10, le=60)
    children: int = Field(..., ge=0)
    smoker: str = Field(..., pattern='^(yes|no)$')
    region: str = Field(..., pattern='^(northeast|northwest|southeast|southwest)$')

class PredictionResponse(BaseModel):
    linearRegression: float
    randomForest: float
    gradientBoosting: float
    xgboost: float
    lightgbm: float
    ensemble: float
    inputData: dict
    modelAccuracies: dict

def preprocess_data(data):
    df = data.copy()
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_smoker'] = df['age'] * (df['smoker'] == 'yes').astype(int)
    df['bmi_smoker'] = df['bmi'] * (df['smoker'] == 'yes').astype(int)
    df['age_children'] = df['age'] * df['children']
    df['bmi_children'] = df['bmi'] * df['children']
    df['age2'] = df['age'] ** 2
    df['age3'] = df['age'] ** 3
    df['bmi2'] = df['bmi'] ** 2
    df['bmi3'] = df['bmi'] ** 3
    df['log_age'] = np.log1p(df['age'])
    df['log_bmi'] = np.log1p(df['bmi'])
    df['is_smoker'] = (df['smoker'] == 'yes').astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['has_children'] = (df['children'] > 0).astype(int)
    df['is_senior'] = (df['age'] >= 50).astype(int)
    df['smoker_obese'] = df['is_smoker'] * df['is_obese']
    df['senior_smoker'] = df['is_senior'] * df['is_smoker']
    df['senior_obese'] = df['is_senior'] * df['is_obese']
    df['risk_score'] = df['is_smoker'] * 3 + df['is_obese'] * 2 + df['is_senior'] * 1.5 + df['children'] * 0.5
    return df

def calculate_accuracy_percentage(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = max(0, 100 - mape)
    return round(accuracy, 2)

def train_models():
    try:
        print('Loading and preprocessing data...')
        data = pd.read_csv('insurance.csv')
        data = preprocess_data(data)
        X = data.drop('charges', axis=1)
        y = data['charges']
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')
        preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
        
        print('Training models...')
        print('[1/6] Training Ridge Regression...')
        lr_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge(alpha=10.0))])
        lr_pipeline.fit(X_train, y_train)
        trained_models['lr'] = lr_pipeline
        lr_pred = lr_pipeline.predict(X_test)
        model_metrics['lr'] = {'r2': round(r2_score(y_test, lr_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2), 'mae': round(mean_absolute_error(y_test, lr_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, lr_pred)}
        print(f"  R2: {model_metrics['lr']['r2']}%, Accuracy: {model_metrics['lr']['accuracy']}%")
        
        print('[2/6] Training Random Forest...')
        rf_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1))])
        rf_pipeline.fit(X_train, y_train)
        trained_models['rf'] = rf_pipeline
        rf_pred = rf_pipeline.predict(X_test)
        model_metrics['rf'] = {'r2': round(r2_score(y_test, rf_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2), 'mae': round(mean_absolute_error(y_test, rf_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, rf_pred)}
        print(f"  R2: {model_metrics['rf']['r2']}%, Accuracy: {model_metrics['rf']['accuracy']}%")
        
        print('[3/6] Training Gradient Boosting...')
        gb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42))])
        gb_pipeline.fit(X_train, y_train)
        trained_models['gb'] = gb_pipeline
        gb_pred = gb_pipeline.predict(X_test)
        model_metrics['gb'] = {'r2': round(r2_score(y_test, gb_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, gb_pred)), 2), 'mae': round(mean_absolute_error(y_test, gb_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, gb_pred)}
        print(f"  R2: {model_metrics['gb']['r2']}%, Accuracy: {model_metrics['gb']['accuracy']}%")
        
        print('[4/6] Training XGBoost...')
        xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1))])
        xgb_pipeline.fit(X_train, y_train)
        trained_models['xgb'] = xgb_pipeline
        xgb_pred = xgb_pipeline.predict(X_test)
        model_metrics['xgb'] = {'r2': round(r2_score(y_test, xgb_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, xgb_pred)), 2), 'mae': round(mean_absolute_error(y_test, xgb_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, xgb_pred)}
        print(f"  R2: {model_metrics['xgb']['r2']}%, Accuracy: {model_metrics['xgb']['accuracy']}%")
        
        print('[5/6] Training LightGBM...')
        lgb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1))])
        lgb_pipeline.fit(X_train, y_train)
        trained_models['lgb'] = lgb_pipeline
        lgb_pred = lgb_pipeline.predict(X_test)
        model_metrics['lgb'] = {'r2': round(r2_score(y_test, lgb_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, lgb_pred)), 2), 'mae': round(mean_absolute_error(y_test, lgb_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, lgb_pred)}
        print(f"  R2: {model_metrics['lgb']['r2']}%, Accuracy: {model_metrics['lgb']['accuracy']}%")
        
        print('[6/6] Training Ensemble Stacking Model...')
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        base_models = [('rf', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)), ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)), ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1))]
        stacking = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5)
        stacking.fit(X_train_preprocessed, y_train)
        
        class StackingPipeline:
            def __init__(self, preprocessor, stacking):
                self.preprocessor = preprocessor
                self.stacking = stacking
            def predict(self, X):
                return self.stacking.predict(self.preprocessor.transform(X))
        
        trained_models['ensemble'] = StackingPipeline(preprocessor, stacking)
        ensemble_pred = stacking.predict(X_test_preprocessed)
        model_metrics['ensemble'] = {'r2': round(r2_score(y_test, ensemble_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, ensemble_pred)), 2), 'mae': round(mean_absolute_error(y_test, ensemble_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, ensemble_pred)}
        print(f"  R2: {model_metrics['ensemble']['r2']}%, Accuracy: {model_metrics['ensemble']['accuracy']}%")
        
        print('='*60)
        print('ALL MODELS TRAINED SUCCESSFULLY!')
        print('='*60)
        print('MODEL PERFORMANCE SUMMARY:')
        for model_name, metrics in model_metrics.items():
            acc_str = str(metrics['accuracy'])
            r2_str = str(metrics['r2'])
            rmse_str = str(metrics['rmse'])
            print(f"{model_name.upper()}: Accuracy: {acc_str}% | R2: {r2_str}% | RMSE: ${rmse_str}")
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        print(f"BEST MODEL: {best_model[0].upper()} with {best_model[1]['accuracy']}% accuracy")
        return True
    except Exception as e:
        print(f'Error training models: {e}')
        import traceback
        traceback.print_exc()
        return False

@app.get('/')
async def root():
    return {'message': 'Medical Insurance Cost Predictor API - Enhanced Edition', 'status': 'running', 'models_loaded': all(model is not None for model in trained_models.values()), 'total_models': len(trained_models)}

@app.get('/accuracy')
async def get_accuracy():
    if not model_metrics:
        raise HTTPException(status_code=503, detail='Models not yet trained. Please try again in a moment.')
    return {'metrics': model_metrics, 'best_model': max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0], 'summary': {'average_accuracy': round(sum(m['accuracy'] for m in model_metrics.values()) / len(model_metrics), 2), 'average_r2': round(sum(m['r2'] for m in model_metrics.values()) / len(model_metrics), 2), 'total_models': len(model_metrics)}}

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: InsuranceRequest):
    if not all(model is not None for model in trained_models.values()):
        raise HTTPException(status_code=503, detail='Models not yet trained. Please try again in a moment.')
    try:
        customer_data = pd.DataFrame({'age': [request.age], 'sex': [request.sex], 'bmi': [request.bmi], 'children': [request.children], 'smoker': [request.smoker], 'region': [request.region]})
        customer_data = preprocess_data(customer_data)
        lr_pred = max(0, float(trained_models['lr'].predict(customer_data)[0]))
        rf_pred = max(0, float(trained_models['rf'].predict(customer_data)[0]))
        gb_pred = max(0, float(trained_models['gb'].predict(customer_data)[0]))
        xgb_pred = max(0, float(trained_models['xgb'].predict(customer_data)[0]))
        lgb_pred = max(0, float(trained_models['lgb'].predict(customer_data)[0]))
        ensemble_pred = max(0, float(trained_models['ensemble'].predict(customer_data)[0]))
        
        return PredictionResponse(
            linearRegression=round(lr_pred, 2), 
            randomForest=round(rf_pred, 2), 
            gradientBoosting=round(gb_pred, 2), 
            xgboost=round(xgb_pred, 2), 
            lightgbm=round(lgb_pred, 2), 
            ensemble=round(ensemble_pred, 2), 
            inputData=request.dict(), 
            modelAccuracies={
                'lr': model_metrics['lr']['accuracy'], 
                'rf': model_metrics['rf']['accuracy'], 
                'gb': model_metrics['gb']['accuracy'], 
                'xgb': model_metrics['xgb']['accuracy'], 
                'lgb': model_metrics['lgb']['accuracy'], 
                'ensemble': model_metrics['ensemble']['accuracy']
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail='Prediction failed: ' + str(e))

@app.get('/models/info')
async def models_info():
    return {'models': {'ridge_regression': trained_models['lr'] is not None, 'random_forest': trained_models['rf'] is not None, 'gradient_boosting': trained_models['gb'] is not None, 'xgboost': trained_models['xgb'] is not None, 'lightgbm': trained_models['lgb'] is not None, 'ensemble_stacking': trained_models['ensemble'] is not None}, 'all_loaded': all(model is not None for model in trained_models.values()), 'performance': model_metrics if model_metrics else 'Not yet calculated'}

if __name__ == '__main__':
    import uvicorn
    if not os.path.exists('insurance.csv'):
        print('Warning: insurance.csv not found!')
    print('Medical Insurance Cost Predictor API - ENHANCED EDITION')
    print('Server will run at: http://localhost:8000')
    print('API Docs: http://localhost:8000/docs')
    print('Accuracy Endpoint: http://localhost:8000/accuracy')
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
