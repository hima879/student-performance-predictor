# model.py
# Train the Random Forest model with hyperparameter tuning
# UPDATED: Custom class weights, better grid, XGBoost support

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our data preparation function
from data_prep import load_and_prepare_data

def train_and_evaluate():
    """Train Random Forest and save the model with hyperparameter tuning"""
    
    print("="*50)
    print("🎯 STUDENT PERFORMANCE PREDICTOR - TRAINING")
    print("="*50)
    
    # Load prepared data
    X, y, feature_names = load_and_prepare_data()
    
    if X is None:
        print("Failed to load data!")
        return None, None

    print(f"\n Dataset shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n Data split:")
    print(f"   Training: {len(X_train)} students")
    print(f"   Testing: {len(X_test)} students")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ========== FEATURE SELECTION ==========
    print("\n🔍 Performing feature selection...")

    # Use a simple model to select important features
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train_scaled, y_train)

    # Get feature importances
    importances = selector_model.feature_importances_

    # ===== CHANGE 1: LOWER THRESHOLD TO KEEP MORE FEATURES =====
    # Select features that are at least 0.5% important (was 1.0%)
    threshold = 0.005  # 0.5%
    selected_features = np.where(importances >= threshold)[0]
    print(f"    Selected {len(selected_features)} out of {len(feature_names)} features")
    print(f"    (Threshold: {threshold*100}% importance)")

    # Reduce data to selected features
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    selected_feature_names = [feature_names[i] for i in selected_features]

    # ========== OPTIMIZED HYPERPARAMETER TUNING ==========
    print("\n🔍 Tuning hyperparameters...")
    
    # ===== CHANGE 2: BETTER PARAMETER GRID WITH CUSTOM CLASS WEIGHTS =====
    param_grid = {
        'n_estimators': [200, 300, 400],  # Added 400
        'max_depth': [10, 12, 15, 20],    # Added 20
        'min_samples_split': [2, 4, 6],   # Added 6
        'min_samples_leaf': [1, 2, 4],    # Added 4
        'max_features': ['sqrt', 'log2'],
        'class_weight': [
            {0: 1.0, 1: 2.0, 2: 1.0},    # 2x weight for Average
            {0: 1.0, 1: 2.5, 2: 1.0},    # 2.5x weight for Average
            {0: 1.0, 1: 3.0, 2: 1.0},    # 3x weight for Average
            'balanced',
            'balanced_subsample'
        ]
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    model = grid_search.best_estimator_
    print(f"\n   ✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")
    
    # ========== EVALUATION ==========
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Test Accuracy: {accuracy*100:.2f}%")
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"📈 Cross-validation Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    print(f"\n📊 Classification Report:")
    report = classification_report(y_test, y_pred, 
                                   target_names=['At-Risk', 'Average', 'Good'],
                                   output_dict=True)
    
    for class_name in ['At-Risk', 'Average', 'Good']:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"      Precision: {metrics['precision']:.2f}")
            print(f"      Recall: {metrics['recall']:.2f}")
            print(f"      F1-Score: {metrics['f1-score']:.2f}")
    
    print(f"\n   Overall Accuracy: {report['accuracy']*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   {cm}")

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f"\n📊 Confusion Matrix (%):")
    for i, label in enumerate(['At-Risk', 'Average', 'Good']):
        print(f"   {label}: {cm_percent[i].round(1)}%")
    
    # ========== VISUALIZATIONS ==========
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['At-Risk', 'Average', 'Good'],
                yticklabels=['At-Risk', 'Average', 'Good'])
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy*100:.1f}%', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("\n   ✅ Saved: confusion_matrix.png")
    
    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh([feature_names[i] for i in sorted_idx[:15]], 
             importances[sorted_idx[:15]])
    plt.title('Top 15 Feature Importances', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("   ✅ Saved: feature_importance.png")
    
    # ========== SAVE MODEL ==========
    joblib.dump(model, 'model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\n💾 Model saved as: model.pkl")
    print(f"💾 Feature names saved as: feature_names.pkl")
    print(f"💾 Scaler saved as: scaler.pkl")
    
    print(f"\n📋 Model Summary:")
    print(f"   Type: Random Forest Classifier")
    print(f"   Trees: {model.n_estimators}")
    print(f"   Max Depth: {model.max_depth}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Selected Features: {len(selected_feature_names)}")
    print(f"   Classes: {model.n_classes_}")

    print(f"\n📊 Top 10 Most Important Features:")
    top_10_idx = sorted_idx[:10]
    for i, idx in enumerate(top_10_idx, 1):
        feature = feature_names[idx]
        importance = importances[idx]
        print(f"   {i}. {feature}: {importance:.3f}")
    
    # ===== CHANGE 3: TRY XGBOOST FOR COMPARISON =====
    print("\n" + "="*50)
    print("🔍 TRYING XGBOOST FOR COMPARISON")
    print("="*50)
    
    try:
        from xgboost import XGBClassifier
        
        print("   Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2,  # Focus on Average class
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train_scaled, y_train)
        xgb_pred = xgb.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"\n   ✅ XGBoost Test Accuracy: {xgb_acc*100:.2f}%")
        
        # If XGBoost is better, save it too
        if xgb_acc > accuracy:
            print("   🎯 XGBoost performs better! Consider using it.")
            joblib.dump(xgb, 'xgb_model.pkl')
            print("   💾 XGBoost model saved as: xgb_model.pkl")
        else:
            print("   ℹ️ Random Forest performs better for this dataset.")
            
    except ImportError:
        print("   ℹ️ XGBoost not installed. Install with: pip install xgboost")
    
    return model, feature_names

if __name__ == "__main__":
    model, features = train_and_evaluate()
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("=" * 50)
    print("\n🚀 Next step: Run 'streamlit run app.py'")