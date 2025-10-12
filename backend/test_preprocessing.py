import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats
        from sklearn.preprocessing import RobustScaler, OneHotEncoder
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    print("\nTesting data loading...")
    try:
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), 'insurance.csv')
        df = pd.read_csv(csv_path)
        print(f"✅ Data loaded: {len(df)} records")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_preprocessing_functions():
    print("\nTesting preprocessing functions...")
    try:
        from app import clean_data, preprocess_data
        import pandas as pd
        
        csv_path = os.path.join(os.path.dirname(__file__), 'insurance.csv')
        df = pd.read_csv(csv_path)
        
        print("  Testing clean_data()...")
        cleaned = clean_data(df)
        print(f"  ✅ Cleaned data: {len(cleaned)} records (from {len(df)})")
        
        print("  Testing preprocess_data()...")
        preprocessed = preprocess_data(cleaned)
        print(f"  ✅ Preprocessed data: {len(preprocessed.columns)} features")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("PREPROCESSING FUNCTIONALITY TEST")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Preprocessing Functions", test_preprocessing_functions()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your preprocessing implementation is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please check the error messages above.")
    print("="*70)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
