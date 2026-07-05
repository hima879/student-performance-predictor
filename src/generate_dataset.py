"""
generate_dataset.py - Synthetic Dataset Generator for Student Performance Prediction

This module creates a realistic dataset of 1000 students with 8 features and a
target variable (performance_level). The data generation process includes:
1. Realistic non-uniform distributions for each feature
2. Non-linear relationships between features and performance
3. Edge cases to prevent trivial pattern learning
4. Controlled label noise

Features (8 total):
1. study_hours (0.5-10) - Average hours studied per day
2. attendance (35-100) - Class attendance percentage
3. previous_score (28-100) - Score on previous exam
4. assignments_completed (0-10) - Assignments completed out of 10
5. sleep_hours (3-10) - Average sleep per night
6. parental_support (Low/Medium/High) - Parental support level
7. extra_curricular (Yes/No) - Participates in extracurriculars
8. internet_access (Yes/No) - Has internet access at home

TARGET: performance_level (At-Risk / Average / Good)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Constants
N_STUDENTS = 1000
N_EDGE_CASES = 40
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


# ============================================================================
# PART 1: GENERATE BASE FEATURES
# ============================================================================
def generate_base_features(n: int = N_STUDENTS) -> pd.DataFrame:
    """
    Generate realistic base features for n students.
    
    Each feature uses a different distribution to mimic real-world student data:
    - study_hours: Right-skewed (most students study 3-6 hours)
    - attendance: Left-skewed (most students attend 80-95%)
    - previous_score: Mixture distribution (70% high, 30% low performers)
    - assignments_completed: Right-skewed (most complete 7-10)
    - sleep_hours: Normal distribution centered around 7-8 hours
    - parental_support: Categorical (Low/Medium/High with varying probabilities)
    - extra_curricular: Categorical (Yes/No with 60/40 split)
    - internet_access: Categorical (Yes/No with 80/20 split)
    
    Args:
        n: Number of student records to generate
        
    Returns:
        DataFrame with raw features for all students
    """
    
    # ----- 1. STUDY HOURS (0.5 - 10 hours) -----
    # Beta distribution gives right-skewed shape
    # Most students study 3-7 hours, some study more
    study_hours = 0.5 + 9.5 * np.random.beta(a=2.5, b=3.5, size=n)
    
    # ----- 2. ATTENDANCE (35% - 100%) -----
    # Beta distribution with left skew (most students have high attendance)
    # Minimum 35% ensures some variation without extreme outliers
    attendance = 35 + 65 * np.random.beta(a=5, b=1.5, size=n)
    
    # ----- 3. PREVIOUS SCORE (28% - 100%) -----
    # Mixture distribution: 70% high-performing, 30% lower-performing
    # This creates realistic bimodal patterns in student performance
    prev_score = np.zeros(n)
    for i in range(n):
        if np.random.random() < 0.7:    # High performing group (70%)
            prev_score[i] = 60 + 40 * np.random.beta(a=3, b=2)
        else:   # Low performing group (30%)
            prev_score[i] = 28 + 32 * np.random.beta(a=2, b=3)
    prev_score = np.clip(prev_score, 28, 100)
    
    # ----- 4. ASSIGNMENTS COMPLETED (0 - 10) -----
    # Right-skewed: most students complete 7-10 assignments
    assignments = np.clip(
        np.random.beta(a=3.5, b=1.2, size=n) * 10,
        0, 10
    )
    assignments = np.round(assignments).astype(int)
    
    # ----- 5. SLEEP HOURS (3 - 10 hours) -----
    # Normal distribution centered at 7.5 hours (optimal sleep)
    sleep_hours = np.random.normal(loc=7.5, scale=1.2, size=n)
    sleep_hours = np.clip(sleep_hours, 3, 10)
    
    # ----- 6. PARENTAL SUPPORT (Categorical) -----
    # 20% Low, 50% Medium, 30% High
    parental_support = np.random.choice(
        ['Low', 'Medium', 'High'], 
        size=n, 
        p=[0.20, 0.50, 0.30]
    )
    
    # ----- 7. EXTRA CURRICULAR (Categorical) -----
    # 60% Yes, 40% No
    extra_curricular = np.random.choice(
        ['Yes', 'No'], 
        size=n, 
        p=[0.60, 0.40]
    )
    
    # ----- 8. INTERNET ACCESS (Categorical) -----
    # 80% Yes, 20% No
    internet_access = np.random.choice(
        ['Yes', 'No'], 
        size=n, 
        p=[0.80, 0.20]
    )
    
    # Combine all features into DataFrame
    df = pd.DataFrame({
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 1),
        'previous_score': np.round(prev_score, 1),
        'assignments_completed': assignments,
        'sleep_hours': np.round(sleep_hours, 1),
        'parental_support': parental_support,
        'extra_curricular': extra_curricular,
        'internet_access': internet_access
    })
    
    return df


# ============================================================================
# PART 2: COMPUTE LATENT ABILITY SCORE
# ============================================================================
def _normalize(series: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a series to [0, 1] range.
    
    Args:
        series: Input array
        
    Returns:
        Normalized array (0-1 range)
    """
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def compute_latent_ability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a hidden 'true ability' score that determines performance.
    
    NON-LINEAR RELATIONSHIPS baked in on purpose:
    - study_hours uses sqrt() -> diminishing returns
    - sleep has a PENALTY band (inverted-U effect)
    - interaction between attendance and previous_score
    - Each feature contributes with different weights
    
    Args:
        df: DataFrame with base features
        
    Returns:
        Array of latent ability scores (continuous 0-1 range)
    """
    
    # Convert categorical to numeric
    support_num = df["parental_support"].map({"Low": 0, "Medium": 1, "High": 2})
    extra_num = (df["extra_curricular"] == "Yes").astype(int)
    internet_num = (df["internet_access"] == "Yes").astype(int)
    
    # ----- COMPONENT 1: STUDY HOURS (Diminishing returns) -----
    # sqrt transformation means each additional hour gives less benefit
    study_effect = _normalize(np.sqrt(df["study_hours"]))
    
    # ----- COMPONENT 2: ATTENDANCE (Linear positive) -----
    attendance_effect = _normalize(df["attendance"])
    
    # ----- COMPONENT 3: PREVIOUS SCORE (Linear positive) -----
    score_effect = _normalize(df["previous_score"])
    
    # ----- COMPONENT 4: ASSIGNMENTS (Power transformation) -----
    assignment_effect = _normalize(df["assignments_completed"])
    
    # ----- COMPONENT 5: PARENTAL SUPPORT -----
    support_effect = support_num / 2.0  # 0, 0.5, 1.0
    
    # ----- COMPONENT 6: SLEEP (Inverted-U effect) -----
    # Optimal at 7.5 hours, penalty for too little OR too much
    sleep_penalty = -0.06 * (df["sleep_hours"] - 7.5) ** 2
    sleep_effect = _normalize(sleep_penalty)
    
    # ----- COMPONENT 7: INTERACTION (Attendance × Previous Score) -----
    # Students with high attendance benefit more from prior knowledge
    interaction = _normalize(attendance_effect * score_effect)
    
    # ----- COMPONENT 8: EXTRACURRICULAR + INTERNET (Bonus) -----
    # Both slightly boost performance
    extra_internet_effect = 0.5 * extra_num + 0.5 * internet_num
    
    # ----- WEIGHTED SUM -----
    latent = (
        0.23 * score_effect +           # Previous score: 23% weight
        0.18 * study_effect +           # Study hours: 18% weight
        0.17 * attendance_effect +      # Attendance: 17% weight
        0.13 * assignment_effect +      # Assignments: 13% weight
        0.09 * support_effect +         # Parental support: 9% weight
        0.08 * sleep_effect +           # Sleep: 8% weight
        0.07 * interaction +            # Interaction: 7% weight
        0.05 * extra_internet_effect    # Extra + Internet: 5% weight
    )
    
    # ----- ADD GAUSSIAN NOISE -----
    # No real-world process is perfectly deterministic
    # This keeps the classification problem realistic
    # Expect ~85-90% accuracy ceiling
    noise = np.random.normal(0, 0.07, len(df))
    latent = latent + noise
    
    # Clip to valid range and re-normalize
    latent = np.clip(latent, 0, 1)
    latent = _normalize(latent)
    
    return latent


# ============================================================================
# PART 3: ASSIGN PERFORMANCE LABELS
# ============================================================================
def assign_performance_level(latent: np.ndarray) -> np.ndarray:
    """
    Convert continuous latent score into 3 classes using QUANTILE thresholds.
    
    Distribution targets:
    - Bottom 30% -> At-Risk
    - Middle 45% -> Average
    - Top 25% -> Good
    
    Using quantiles ensures class balance stays sensible no matter how 
    the random noise lands.
    
    Args:
        latent: Array of continuous ability scores
        
    Returns:
        Array of categorical labels
    """
    
    # Calculate quantile thresholds
    q_low = np.quantile(latent, 0.30)    # 30th percentile
    q_high = np.quantile(latent, 0.75)   # 75th percentile
    
    # Assign labels
    labels = np.where(
        latent < q_low, 
        "At-Risk", 
        np.where(latent < q_high, "Average", "Good")
    )
    
    return labels


# ============================================================================
# PART 4: INJECT EDGE CASES
# ============================================================================
def inject_edge_cases(df: pd.DataFrame, labels: np.ndarray, 
                      n_edge: int = N_EDGE_CASES) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Deliberately inject surprising students so the model can't just memorize 
    simple rules like 'more study hours = better'.
    
    Edge case types (4 types, ~10 each):
    1. 'Gifted but low-effort' - low study hours, high previous score
    2. 'Hard-working but overwhelmed' - high study/attendance, poor sleep
    3. 'Quiet achiever' - low attendance but strong self-study
    4. 'Label noise' - random/contradictory rows (real-world data is never clean)
    
    Args:
        df: Base features DataFrame
        labels: Performance labels array
        n_edge: Number of edge cases to inject
        
    Returns:
        Tuple of (updated DataFrame, updated labels)
    """
    
    # Make copies to modify
    df_modified = df.copy()
    labels_modified = labels.copy()
    
    # Select indices to replace
    indices_to_replace = np.random.choice(df.index, size=n_edge, replace=False)
    chunks = np.array_split(indices_to_replace, 4)
    
    # ----- TYPE 1: GIFTED BUT LOW-EFFORT (~10 cases) -----
    # Low study hours, high previous score, decent attendance -> Good
    for i in chunks[0]:
        df_modified.loc[i, "study_hours"] = np.random.uniform(0.5, 1.8)
        df_modified.loc[i, "previous_score"] = np.random.uniform(85, 98)
        df_modified.loc[i, "attendance"] = np.random.uniform(70, 90)
        df_modified.loc[i, "sleep_hours"] = np.random.uniform(6, 8)
        df_modified.loc[i, "assignments_completed"] = np.random.randint(6, 9)
        labels_modified[df.index.get_loc(i)] = "Good"
    
    # ----- TYPE 2: HARD-WORKING BUT OVERWHELMED (~10 cases) -----
    # High study and attendance, but poor sleep and low previous score
    for i in chunks[1]:
        df_modified.loc[i, "study_hours"] = np.random.uniform(7.5, 10)
        df_modified.loc[i, "attendance"] = np.random.uniform(85, 100)
        df_modified.loc[i, "sleep_hours"] = np.random.uniform(3, 4.5)
        df_modified.loc[i, "previous_score"] = np.random.uniform(30, 48)
        df_modified.loc[i, "assignments_completed"] = np.random.randint(7, 10)
        labels_modified[df.index.get_loc(i)] = np.random.choice(["At-Risk", "Average"])
    
    # ----- TYPE 3: QUIET ACHIEVER (~10 cases) -----
    # Low attendance, but strong self-study and high previous score
    for i in chunks[2]:
        df_modified.loc[i, "attendance"] = np.random.uniform(40, 58)
        df_modified.loc[i, "previous_score"] = np.random.uniform(80, 95)
        df_modified.loc[i, "study_hours"] = np.random.uniform(5, 7)
        df_modified.loc[i, "sleep_hours"] = np.random.uniform(7, 9)
        df_modified.loc[i, "assignments_completed"] = np.random.randint(7, 10)
        labels_modified[df.index.get_loc(i)] = np.random.choice(["Average", "Good"])
    
    # ----- TYPE 4: PURE LABEL NOISE (~10 cases) -----
    # Random re-labeling, simulates real-world grading inconsistency
    for i in chunks[3]:
        labels_modified[df.index.get_loc(i)] = np.random.choice(
            ["At-Risk", "Average", "Good"]
        )
    
    return df_modified, labels_modified


# ============================================================================
# PART 5: MAIN DATASET GENERATION PIPELINE
# ============================================================================
def generate_dataset(n: int = N_STUDENTS, n_edge: int = N_EDGE_CASES) -> pd.DataFrame:
    """
    Complete dataset generation pipeline.
    
    Steps:
    1. Generate base features with realistic distributions (8 features)
    2. Compute latent ability score (non-linear relationships)
    3. Assign performance labels (quantile-based thresholds)
    4. Inject edge cases (40 anomalies)
    
    Args:
        n: Number of students to generate
        n_edge: Number of edge cases to inject
        
    Returns:
        Complete DataFrame with features and target label
    """
    
    print(f"Generating dataset with {n} students, {n_edge} edge cases...")
    
    # Step 1: Generate base features
    print("  → Generating 8 base features with realistic distributions...")
    df = generate_base_features(n)
    
    # Step 2: Compute latent ability
    print("  → Computing latent ability score (non-linear relationships)...")
    latent = compute_latent_ability(df)
    
    # Step 3: Assign labels
    print("  → Assigning performance levels (quantile-based thresholds)...")
    labels = assign_performance_level(latent)
    
    # Step 4: Inject edge cases
    if n_edge > 0:
        print(f"  → Injecting {n_edge} edge cases (4 types of anomalies)...")
        df, labels = inject_edge_cases(df, labels, n_edge)
    
    # Step 5: Add target column
    df["performance_level"] = labels
    
    return df


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    dataset = generate_dataset()
    
    # Save to CSV
    out_path = data_dir / "dataset.csv"
    dataset.to_csv(out_path, index=False)
    
    # Print statistics
    print(f"\n✅ Dataset generated: {out_path} ({len(dataset)} rows, {len(dataset.columns)} columns)")
    print("\n📊 Class distribution:")
    print(dataset["performance_level"].value_counts())
    print("\n📋 Sample rows (first 5):")
    print(dataset.head(10).to_string(index=False))
    
    print("\n📈 Feature statistics:")
    numeric_cols = ['study_hours', 'attendance', 'previous_score', 
                    'assignments_completed', 'sleep_hours']
    for col in numeric_cols:
        print(f"  • {col}: min={dataset[col].min():.1f}, "
              f"mean={dataset[col].mean():.1f}, "
              f"max={dataset[col].max():.1f}")
    
    print("\n🏷️ Categorical feature distributions:")
    cat_cols = ['parental_support', 'extra_curricular', 'internet_access']
    for col in cat_cols:
        print(f"  • {col}:")
        for val, count in dataset[col].value_counts().items():
            print(f"      {val}: {count} ({count/len(dataset)*100:.1f}%)")