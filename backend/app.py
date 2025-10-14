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
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score
from scipy import stats
import lightgbm as lgb
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

trained_models = {'rf': None, 'gb': None, 'lgb': None, 'ensemble': None}
model_metrics = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting up FastAPI server...')
    train_models()
    yield
    print('Shutting down FastAPI server...')

app = FastAPI(title='MediPredict API - Enhanced', lifespan=lifespan)

cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:5174').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins],
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
    ensemble: float
    inputData: dict
    modelAccuracies: dict

def clean_data(data):
    df = data.copy()
    
    initial_rows = len(df)
    df = df.drop_duplicates(keep='first')
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f'  Removed {duplicates_removed} duplicate records')
    
    bmi_outliers_low = len(df[df['bmi'] < 15])
    bmi_outliers_high = len(df[df['bmi'] > 50])
    
    if bmi_outliers_low > 0:
        print(f'  Correcting {bmi_outliers_low} extremely low BMI values')
        df.loc[df['bmi'] < 15, 'bmi'] = 15.0
    
    if bmi_outliers_high > 0:
        print(f'  Capping {bmi_outliers_high} extremely high BMI values')
        df.loc[df['bmi'] > 50, 'bmi'] = 50.0
    
    Q1 = df['charges'].quantile(0.25)
    Q3 = df['charges'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    charges_outliers_low = len(df[df['charges'] < lower_bound])
    charges_outliers_high = len(df[df['charges'] > upper_bound])
    
    if charges_outliers_low > 0:
        print(f'  Adjusting {charges_outliers_low} unusually low charge values')
        df.loc[df['charges'] < lower_bound, 'charges'] = lower_bound
    
    if charges_outliers_high > 0:
        print(f'  Capping {charges_outliers_high} extremely high charge values')
        df.loc[df['charges'] > upper_bound, 'charges'] = upper_bound
    
    df['sex'] = df['sex'].str.lower().str.strip()
    df['smoker'] = df['smoker'].str.lower().str.strip()
    df['region'] = df['region'].str.lower().str.strip()
    
    age_invalid = len(df[(df['age'] < 18) | (df['age'] > 64)])
    if age_invalid > 0:
        print(f'  Filtering {age_invalid} records with invalid age')
        df = df[(df['age'] >= 18) & (df['age'] <= 64)]
    
    children_invalid = len(df[(df['children'] < 0) | (df['children'] > 5)])
    if children_invalid > 0:
        print(f'  Filtering {children_invalid} records with invalid children count')
        df = df[(df['children'] >= 0) & (df['children'] <= 5)]
    
    print(f'  Final dataset size: {len(df)} records (cleaned from {initial_rows})')
    
    return df.reset_index(drop=True)

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
    df['log_charges'] = np.log1p(df.get('charges', 0))
    
    df['is_smoker'] = (df['smoker'] == 'yes').astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['has_children'] = (df['children'] > 0).astype(int)
    df['is_senior'] = (df['age'] >= 50).astype(int)
    df['is_young'] = (df['age'] < 30).astype(int)
    
    df['smoker_obese'] = df['is_smoker'] * df['is_obese']
    df['senior_smoker'] = df['is_senior'] * df['is_smoker']
    df['senior_obese'] = df['is_senior'] * df['is_obese']
    df['young_smoker'] = df['is_young'] * df['is_smoker']
    
    df['risk_score'] = (
        df['is_smoker'] * 3 + 
        df['is_obese'] * 2 + 
        df['is_senior'] * 1.5 + 
        df['children'] * 0.5
    )
    
    df['bmi_deviation'] = np.abs(df['bmi'] - 25)
    
    return df

def calculate_accuracy_percentage(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = max(0, 100 - mape)
    return round(accuracy, 2)

def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score by converting regression to classification bins.
    Bins are based on quartiles: low, medium-low, medium-high, high cost.
    """
    # Create bins based on quartiles of y_true
    bins = [0, np.percentile(y_true, 25), np.percentile(y_true, 50), 
            np.percentile(y_true, 75), np.inf]
    
    # Convert to categorical labels
    y_true_binned = pd.cut(y_true, bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
    
    # Calculate F1 score with macro averaging
    f1 = f1_score(y_true_binned, y_pred_binned, average='macro', zero_division=0)
    return round(f1, 4)

def train_models():
    try:
        print('='*60)
        print('LOADING AND PREPROCESSING DATA')
        print('='*60)
        
        data = pd.read_csv('insurance.csv')
        print(f'Initial dataset: {len(data)} records')
        
        print('\n[STEP 1] Cleaning data...')
        data = clean_data(data)
        
        print('\n[STEP 2] Creating advanced features...')
        data = preprocess_data(data)
        print(f'  Created {len(data.columns) - 7} additional features')
        
        X = data.drop('charges', axis=1)
        y = data['charges']
        
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f'  Numerical features: {len(numerical_features)}')
        print(f'  Categorical features: {len(categorical_features)}')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        print(f'\n[STEP 3] Data split: {len(X_train)} training, {len(X_test)} testing samples')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        print('\n' + '='*60)
        print('TRAINING MODELS')
        print('='*60)
        print('[1/4] Training Random Forest...')
        rf_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1))])
        rf_pipeline.fit(X_train, y_train)
        trained_models['rf'] = rf_pipeline
        rf_pred = rf_pipeline.predict(X_test)
        model_metrics['rf'] = {'r2': round(r2_score(y_test, rf_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2), 'mae': round(mean_absolute_error(y_test, rf_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, rf_pred), 'f1': calculate_f1_score(y_test, rf_pred)}
        print(f"  R2: {model_metrics['rf']['r2']}%, Accuracy: {model_metrics['rf']['accuracy']}%, F1: {model_metrics['rf']['f1']}")
        
        print('[2/4] Training Gradient Boosting...')
        gb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42))])
        gb_pipeline.fit(X_train, y_train)
        trained_models['gb'] = gb_pipeline
        gb_pred = gb_pipeline.predict(X_test)
        model_metrics['gb'] = {'r2': round(r2_score(y_test, gb_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, gb_pred)), 2), 'mae': round(mean_absolute_error(y_test, gb_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, gb_pred), 'f1': calculate_f1_score(y_test, gb_pred)}
        print(f"  R2: {model_metrics['gb']['r2']}%, Accuracy: {model_metrics['gb']['accuracy']}%, F1: {model_metrics['gb']['f1']}")
        
        print('[3/4] Training LightGBM...')
        lgb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1))])
        lgb_pipeline.fit(X_train, y_train)
        trained_models['lgb'] = lgb_pipeline
        lgb_pred = lgb_pipeline.predict(X_test)
        model_metrics['lgb'] = {'r2': round(r2_score(y_test, lgb_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, lgb_pred)), 2), 'mae': round(mean_absolute_error(y_test, lgb_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, lgb_pred), 'f1': calculate_f1_score(y_test, lgb_pred)}
        print(f"  R2: {model_metrics['lgb']['r2']}%, Accuracy: {model_metrics['lgb']['accuracy']}%, F1: {model_metrics['lgb']['f1']}")
        
        print('[4/4] Training Ensemble Stacking Model...')
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
        model_metrics['ensemble'] = {'r2': round(r2_score(y_test, ensemble_pred) * 100, 2), 'rmse': round(np.sqrt(mean_squared_error(y_test, ensemble_pred)), 2), 'mae': round(mean_absolute_error(y_test, ensemble_pred), 2), 'accuracy': calculate_accuracy_percentage(y_test, ensemble_pred), 'f1': calculate_f1_score(y_test, ensemble_pred)}
        print(f"  R2: {model_metrics['ensemble']['r2']}%, Accuracy: {model_metrics['ensemble']['accuracy']}%, F1: {model_metrics['ensemble']['f1']}")
        
        print('='*60)
        print('ALL MODELS TRAINED SUCCESSFULLY!')
        print('='*60)
        print('MODEL PERFORMANCE SUMMARY:')
        for model_name, metrics in model_metrics.items():
            acc_str = str(metrics['accuracy'])
            r2_str = str(metrics['r2'])
            rmse_str = str(metrics['rmse'])
            f1_str = str(metrics['f1'])
            print(f"{model_name.upper()}: Accuracy: {acc_str}% | R2: {r2_str}% | RMSE: ${rmse_str} | F1: {f1_str}")
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
    return {'metrics': model_metrics, 'best_model': max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0], 'summary': {'average_accuracy': round(sum(m['accuracy'] for m in model_metrics.values()) / len(model_metrics), 2), 'average_r2': round(sum(m['r2'] for m in model_metrics.values()) / len(model_metrics), 2), 'average_f1': round(sum(m['f1'] for m in model_metrics.values()) / len(model_metrics), 4), 'total_models': len(model_metrics)}}

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: InsuranceRequest):
    if not all(model is not None for model in trained_models.values()):
        raise HTTPException(status_code=503, detail='Models not yet trained. Please try again in a moment.')
    try:
        customer_data = pd.DataFrame({'age': [request.age], 'sex': [request.sex], 'bmi': [request.bmi], 'children': [request.children], 'smoker': [request.smoker], 'region': [request.region]})
        customer_data = preprocess_data(customer_data)
        ensemble_pred = max(0, float(trained_models['ensemble'].predict(customer_data)[0]))
        
        return PredictionResponse(
            ensemble=round(ensemble_pred, 2), 
            inputData=request.dict(), 
            modelAccuracies={
                'rf': model_metrics['rf']['accuracy'], 
                'gb': model_metrics['gb']['accuracy'], 
                'lgb': model_metrics['lgb']['accuracy'], 
                'ensemble': model_metrics['ensemble']['accuracy']
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail='Prediction failed: ' + str(e))

@app.get('/models/info')
async def models_info():
    return {'models': {'random_forest': trained_models['rf'] is not None, 'gradient_boosting': trained_models['gb'] is not None, 'lightgbm': trained_models['lgb'] is not None, 'ensemble_stacking': trained_models['ensemble'] is not None}, 'all_loaded': all(model is not None for model in trained_models.values()), 'performance': model_metrics if model_metrics else 'Not yet calculated'}

if __name__ == '__main__':
    import uvicorn
    if not os.path.exists('insurance.csv'):
        print('Warning: insurance.csv not found!')
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    
    print('Medical Insurance Cost Predictor API - ENHANCED EDITION')
    print(f'Server will run at: http://{host}:{port}')
    print(f'API Docs: http://{host}:{port}/docs')
    print(f'Accuracy Endpoint: http://{host}:{port}/accuracy')
    uvicorn.run('app:app', host=host, port=port, reload=True)
