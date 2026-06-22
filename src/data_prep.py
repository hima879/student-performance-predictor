# data_prep.py
# Load and clean the dataset with advanced feature engineering
# UPDATED: Added back critical features + Average class specific features

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load dataset, clean it, and prepare for ML with advanced feature engineering"""
    
    print("📂 Loading dataset...")
    
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"   Current directory: {current_dir}")
    
    # Try to find dataset.csv
    csv_path = os.path.join(current_dir, 'dataset.csv')
    
    if not os.path.exists(csv_path):
        # Try parent directory
        csv_path = os.path.join(os.path.dirname(current_dir), 'dataset.csv')
    
    if not os.path.exists(csv_path):
        # Try src directory (if running from root)
        csv_path = os.path.join(current_dir, 'src', 'dataset.csv')
    
    if not os.path.exists(csv_path):
        print(f"   ❌ dataset.csv not found!")
        print(f"   Tried: {csv_path}")
        print("   💡 Run 'python generate_dataset.py' first")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    
    print(f"   ✅ File found! Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # ========== CHECK FOR EDGE CASES ==========
    print("\n🔍 Analyzing dataset quality...")
    
    # Check for edge cases we added
    high_prev_low_study = ((df['previous_score'] > 80) & (df['study_hours_per_day'] < 4)).sum()
    low_prev_high_study = ((df['previous_score'] < 50) & (df['study_hours_per_day'] > 8)).sum()
    low_sleep_high_perf = ((df['sleep_hours'] < 5) & (df['previous_score'] > 80)).sum()
    
    print(f"   Edge cases found:")
    print(f"      High previous score + Low study: {high_prev_low_study}")
    print(f"      Low previous score + High study: {low_prev_high_study}")
    print(f"      Low sleep + High performance: {low_sleep_high_perf}")
    
    # ========== ADVANCED FEATURE ENGINEERING ==========
    print("\n🔧 Creating new features...")
    
    original_features = len(df.columns) - 2  # minus student_id and target
    new_features_count = 0
    
    # ---- GROUP 1: BASIC FEATURES (No dependencies) ----
    
    # 1. Distance from middle (helps identify Average students)
    df['middle_score'] = (df['previous_score'] - 50).abs()
    new_features_count += 1
    print(f"   ✅ Created: middle_score")
    
    # 2. Borderline indicators (students on the edge)
    df['borderline_at_risk'] = ((df['previous_score'] >= 40) & (df['previous_score'] <= 60)).astype(int)
    new_features_count += 1
    print(f"   ✅ Created: borderline_at_risk")
    
    df['borderline_good'] = ((df['previous_score'] >= 60) & (df['previous_score'] <= 75)).astype(int)
    new_features_count += 1
    print(f"   ✅ Created: borderline_good")
    
    # 3. Study-Attendance Gap (identifies inconsistency)
    df['study_attendance_gap'] = df['study_hours_per_day'] - (df['attendance_percentage'] / 100 * 10)
    new_features_count += 1
    print(f"   ✅ Created: study_attendance_gap")
    
    # 4. Improvement Potential
    df['improvement_potential'] = 100 - df['previous_score']
    new_features_count += 1
    print(f"   ✅ Created: improvement_potential")
    
    # 5. Consistency Score (low variance in habits)
    df['consistency_score'] = (
        (df['study_hours_per_day'] / 12) * 0.4 +
        (df['attendance_percentage'] / 100) * 0.4 +
        (df['sleep_hours'] / 10) * 0.2
    )
    new_features_count += 1
    print(f"   ✅ Created: consistency_score")
    
    # 6. Risk-Reward Ratio
    df['risk_reward'] = df['previous_score'] / (df['study_hours_per_day'] + 1)
    new_features_count += 1
    print(f"   ✅ Created: risk_reward")
    
    # ---- GROUP 2: CRITICAL FEATURES (RESTORED FROM ORIGINAL) ----
    
    # 7. Study-Attendance Interaction (was #2 most important!)
    df['study_attendance'] = df['study_hours_per_day'] * (df['attendance_percentage'] / 100)
    new_features_count += 1
    print(f"   ✅ Created: study_attendance")
    
    # 8. Performance Score (was #5 most important!)
    df['performance_score'] = (
        df['previous_score'] * 0.4 +
        df['study_hours_per_day'] * 0.3 +
        (df['attendance_percentage'] / 100) * 20 * 0.2 +
        df['assignments_completed'] * 0.1
    )
    new_features_count += 1
    print(f"   ✅ Created: performance_score")
    
    # 9. Efficiency Score
    df['efficiency'] = df['assignments_completed'] / df['study_hours_per_day']
    df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0)
    df['efficiency'] = df['efficiency'].fillna(0)
    new_features_count += 1
    print(f"   ✅ Created: efficiency")
    
    # ---- GROUP 3: FEATURES DEPENDING ON BASIC FEATURES ----
    
    # 10. Inconsistency Score (how inconsistent are their habits?)
    df['inconsistency'] = (
        (df['study_hours_per_day'] - df['study_hours_per_day'].mean()).abs() +
        (df['attendance_percentage'] - df['attendance_percentage'].mean()).abs()
    )
    new_features_count += 1
    print(f"   ✅ Created: inconsistency")
    
    # 11. Risk Indicators (MUST BE CREATED BEFORE improvement_risk_ratio!)
    df['at_risk_signals'] = (
        (df['study_hours_per_day'] < 3).astype(int) +
        (df['attendance_percentage'] < 60).astype(int) +
        (df['previous_score'] < 40).astype(int) +
        (df['assignments_completed'] < 5).astype(int)
    )
    new_features_count += 1
    print(f"   ✅ Created: at_risk_signals")
    
    # 12. Improvement vs Risk Ratio (NOW at_risk_signals exists!)
    df['improvement_risk_ratio'] = (100 - df['previous_score']) / (df['at_risk_signals'] + 1)
    new_features_count += 1
    print(f"   ✅ Created: improvement_risk_ratio")
    
    # ---- GROUP 4: CATEGORICAL FEATURES ----
    
    # 13. Study Category
    df['study_category'] = pd.cut(df['study_hours_per_day'],
                                  bins=[0, 3, 5, 7, 13],
                                  labels=['Very_Low', 'Low', 'Medium', 'High'])
    new_features_count += 1
    print(f"   ✅ Created: study_category")
    
    # 14. Attendance Category
    df['attendance_category'] = pd.cut(df['attendance_percentage'],
                                       bins=[0, 60, 75, 90, 101],
                                       labels=['Low', 'Medium', 'High', 'Very_High'])
    new_features_count += 1
    print(f"   ✅ Created: attendance_category")
    
    # 15. Score Category
    df['score_category'] = pd.cut(df['previous_score'],
                                  bins=[0, 40, 60, 80, 101],
                                  labels=['Low', 'Medium', 'High', 'Very_High'])
    new_features_count += 1
    print(f"   ✅ Created: score_category")
    
    # 16. Sleep Quality
    df['sleep_quality'] = np.where(df['sleep_hours'] >= 7, 'Good', 'Poor')
    new_features_count += 1
    print(f"   ✅ Created: sleep_quality")
    
    # 17. Effort Score
    df['effort_score'] = (
        (df['study_hours_per_day'] / 12) * 0.3 +
        (df['assignments_completed'] / 20) * 0.3 +
        (df['attendance_percentage'] / 100) * 0.4
    )
    new_features_count += 1
    print(f"   ✅ Created: effort_score")
    
    # 18. Parental Support Numeric
    df['parental_support_num'] = df['parental_support'].map({'Low': 0, 'Medium': 1, 'High': 2})
    new_features_count += 1
    print(f"   ✅ Created: parental_support_num")
    
    # 19. Support Factors
    df['support_factors'] = (
        (df['extra_curricular'] == 'Yes').astype(int) +
        (df['internet_access'] == 'Yes').astype(int)
    )
    new_features_count += 1
    print(f"   ✅ Created: support_factors")
    
    # 20. Good Sleep Indicator
    df['good_sleep'] = (df['sleep_hours'] >= 7).astype(int)
    new_features_count += 1
    print(f"   ✅ Created: good_sleep")
    
    # 21. Assignment Rate
    df['assignment_rate'] = df['assignments_completed'] / 20
    new_features_count += 1
    print(f"   ✅ Created: assignment_rate")
    
    # ---- GROUP 5: NUMERIC CATEGORY FEATURES ----
    
    # 22. Study Category Numeric
    df['study_category_num'] = pd.cut(df['study_hours_per_day'],
                                      bins=[0, 3, 5, 7, 13],
                                      labels=[0, 1, 2, 3]).astype(float)
    new_features_count += 1
    print(f"   ✅ Created: study_category_num")
    
    # 23. Attendance Category Numeric
    df['attendance_category_num'] = pd.cut(df['attendance_percentage'],
                                           bins=[0, 60, 75, 90, 101],
                                           labels=[0, 1, 2, 3]).astype(float)
    new_features_count += 1
    print(f"   ✅ Created: attendance_category_num")
    
    # 24. Score Category Numeric (FIXED bins!)
    df['score_category_num'] = pd.cut(df['previous_score'],
                                      bins=[0, 40, 60, 80, 101],
                                      labels=[0, 1, 2, 3]).astype(float)
    new_features_count += 1
    print(f"   ✅ Created: score_category_num")
    
    # 25. Sleep Score
    df['sleep_score'] = np.where(df['sleep_hours'] >= 8, 2,
                                 np.where(df['sleep_hours'] >= 6, 1, 0))
    new_features_count += 1
    print(f"   ✅ Created: sleep_score")
    
    # 26. Success Probability
    df['success_prob'] = (
        (df['previous_score'] / 100) * 0.3 +
        (df['study_hours_per_day'] / 12) * 0.2 +
        (df['attendance_percentage'] / 100) * 0.2 +
        (df['assignments_completed'] / 20) * 0.15 +
        (df['sleep_hours'] / 10) * 0.15
    )
    new_features_count += 1
    print(f"   ✅ Created: success_prob")
    
    print(f"\n   Original features: {original_features}")
    print(f"   New features created: {new_features_count}")
    print(f"   Total features before encoding: {len(df.columns) - 2}")
    
    # ========== FEATURE CORRELATION ANALYSIS ==========
    print("\n📊 Analyzing feature correlations with target...")
    
    # Encode target for correlation
    target_encoded = df['performance_level'].map({'At-Risk': 0, 'Average': 1, 'Good': 2})
    
    # Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols]
    
    # Calculate correlations with target
    correlations = numeric_df.corrwith(pd.Series(target_encoded, index=df.index))
    top_correlations = correlations.abs().sort_values(ascending=False)
    
    print("   Top 10 features by correlation with target:")
    for i, (feature, corr) in enumerate(top_correlations.head(10).items(), 1):
        print(f"      {i}. {feature}: {corr:.3f}")
    
    # Identify highly correlated features (may be redundant)
    print("\n   Checking for highly correlated features (>0.9)...")
    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    
    if high_corr:
        print(f"   ⚠️ Found {len(high_corr)} highly correlated features:")
        for feature in high_corr[:5]:  # Show top 5
            print(f"      {feature}")
    else:
        print("   ✅ No highly correlated features found")
    
    # ========== HANDLE MISSING VALUES ==========
    print("\n📊 Checking for missing values...")
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f"   ⚠️ Found {nulls.sum()} missing values - filling them...")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("   ✅ No missing values found")
    
    # ========== REMOVE DUPLICATES ==========
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"   ✅ Removed {initial_rows - len(df)} duplicates")
    else:
        print("   ✅ No duplicates found")
    
    # ========== SPLIT FEATURES AND TARGET ==========
    X = df.drop(['student_id', 'performance_level'], axis=1)
    y = df['performance_level']
    
    # ========== ENCODING ==========
    print("\n🔢 Encoding categorical variables...")
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    print(f"   Categorical columns: {len(categorical_cols)}")
    
    # Convert text to numbers (one-hot encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Convert target to numbers
    y_encoded = y.map({'At-Risk': 0, 'Average': 1, 'Good': 2})
    
    # ========== FINAL SUMMARY ==========
    print(f"\n✅ Data ready!")
    print(f"   Final features: {X_encoded.shape[1]}")
    print(f"   Target classes: {y.value_counts().to_dict()}")
    print(f"   Feature sample: {list(X_encoded.columns)[:5]}...")
    
    return X_encoded, y_encoded, X_encoded.columns.tolist()

# ========== QUICK TEST ==========
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 TESTING DATA PREPARATION")
    print("=" * 50)
    X, y, features = load_and_prepare_data()
    
    if X is not None:
        print("\n✅ Test successful!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Total features: {len(features)}")
        print(f"\n📋 Sample features:")
        for i, feature in enumerate(features[:10]):
            print(f"   {i+1}. {feature}")
        if len(features) > 10:
            print(f"   ... and {len(features) - 10} more")
    else:
        print("\n❌ Test failed - check the error above")