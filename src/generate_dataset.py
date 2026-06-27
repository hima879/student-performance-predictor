# src/generate_dataset.py
"""
generate_dataset.py - Synthetic Dataset with Edge Cases
Generates the 1,000-student dataset with realistic distributions,
non-linear relationships, and injected edge cases.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Constants
N_STUDENTS = 1000
N_EDGE_CASES = 40
RANDOM_SEED = 42


def generate_base_features(n: int) -> pd.DataFrame:
    """
    Generate realistic (non-uniform) distributions for all features.
    Each feature is drawn from a distribution that mimics real student data.
    """
    np.random.seed(RANDOM_SEED)
    
    # Study hours: right-skewed (most students study 2-6 hours, some study more)
    study_hours = np.random.gamma(shape=2.5, scale=1.8, size=n)
    study_hours = np.clip(study_hours, 0.5, 10.0)
    
    # Attendance: mostly high with some low outliers
    attendance = np.random.beta(a=8, b=2, size=n) * 65 + 35
    attendance = np.clip(attendance, 35, 100)
    
    # Previous score: bimodal (some struggling, some doing well)
    prev_score = np.concatenate([
        np.random.normal(45, 12, int(n * 0.3)),
        np.random.normal(78, 12, int(n * 0.7))
    ])
    prev_score = np.clip(prev_score, 28, 100)
    np.random.shuffle(prev_score)
    
    # Assignments completed: correlated with study hours
    assignments_completed = np.random.poisson(lam=3 + study_hours * 0.5, size=n)
    assignments_completed = np.clip(assignments_completed, 0, 10)
    
    # Sleep hours: normal around 6-8 hours
    sleep_hours = np.random.normal(7.0, 1.5, size=n)
    sleep_hours = np.clip(sleep_hours, 3, 10)
    
    # Categorical features
    parental_support = np.random.choice(
        ["Low", "Medium", "High"], 
        size=n, 
        p=[0.25, 0.45, 0.30]
    )
    extra_curricular = np.random.choice(
        ["Yes", "No"], 
        size=n, 
        p=[0.55, 0.45]
    )
    internet_access = np.random.choice(
        ["Yes", "No"], 
        size=n, 
        p=[0.75, 0.25]
    )
    
    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": prev_score,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours,
        "parental_support": parental_support,
        "extra_curricular": extra_curricular,
        "internet_access": internet_access,
    })
    
    return df


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1] range."""
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def compute_latent_ability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a hidden 'true ability' score that determines performance.
    
    NON-LINEAR RELATIONSHIPS baked in on purpose:
    - study_hours uses sqrt() -> diminishing returns
    - sleep has a PENALTY band (inverted-U / threshold effect)
    - interaction between attendance and previous_score
    """
    # Convert categoricals to numeric
    support_num = df["parental_support"].map({"Low": 0, "Medium": 1, "High": 2})
    extra_num = (df["extra_curricular"] == "Yes").astype(int)
    internet_num = (df["internet_access"] == "Yes").astype(int)
    
    # Normalize effects
    study_effect = _normalize(np.sqrt(df["study_hours"]))
    attendance_effect = _normalize(df["attendance"])
    score_effect = _normalize(df["previous_score"])
    assignment_effect = _normalize(df["assignments_completed"])
    support_effect = support_num / 2.0
    
    # Inverted-U sleep effect: ideal is 7-8 hours
    sleep_penalty = -0.06 * (df["sleep_hours"] - 7.5) ** 2
    sleep_effect = _normalize(sleep_penalty)
    
    # Interaction term: attendance compounds with prior performance
    interaction = _normalize(attendance_effect * score_effect)
    
    # Weighted combination
    latent = (
        0.23 * score_effect +
        0.18 * study_effect +
        0.17 * attendance_effect +
        0.13 * assignment_effect +
        0.09 * support_effect +
        0.08 * sleep_effect +
        0.07 * interaction +
        0.05 * (0.5 * extra_num + 0.5 * internet_num)
    )
    
    # Add Gaussian noise for realism (85-90% accuracy ceiling expected)
    noise = np.random.normal(0, 0.07, len(df))
    latent = latent + noise
    
    return latent


def assign_performance_level(latent: np.ndarray) -> np.ndarray:
    """
    Convert continuous latent score into 3 classes using quantile thresholds.
    
    bottom 30% -> At-Risk
    middle 45% -> Average
    top 25% -> Good
    """
    q_low, q_high = np.quantile(latent, [0.30, 0.75])
    labels = np.where(
        latent < q_low, 
        "At-Risk", 
        np.where(latent < q_high, "Average", "Good")
    )
    return labels


def inject_edge_cases(df: pd.DataFrame, labels: np.ndarray, n_edge: int) -> tuple:
    """
    Deliberately inject ~40 'surprising' students so the model can't
    just memorize simple rules like 'more study hours = better'.
    
    Edge case types:
    1. 'Gifted but low-effort' - low study hours, high previous score,
       decent attendance -> still Good.
    2. 'Hard-working but overwhelmed' - high study hours & attendance,
       but poor sleep & low previous score -> stays At-Risk/Average.
    3. 'Quiet achiever' - low attendance but strong self-study and
       high previous score -> Average/Good.
    4. 'Label noise' - random/contradictory rows for realism.
    """
    idx = np.random.choice(df.index, size=n_edge, replace=False)
    chunks = np.array_split(idx, 4)
    
    # Type 1: gifted, low effort
    for i in chunks[0]:
        df.loc[i, "study_hours"] = np.random.uniform(0.5, 1.8)
        df.loc[i, "previous_score"] = np.random.uniform(85, 98)
        df.loc[i, "attendance"] = np.random.uniform(70, 90)
        labels[df.index.get_loc(i)] = "Good"
    
    # Type 2: hard-working but overwhelmed
    for i in chunks[1]:
        df.loc[i, "study_hours"] = np.random.uniform(7.5, 10)
        df.loc[i, "attendance"] = np.random.uniform(85, 100)
        df.loc[i, "sleep_hours"] = np.random.uniform(3, 4.5)
        df.loc[i, "previous_score"] = np.random.uniform(30, 48)
        labels[df.index.get_loc(i)] = np.random.choice(["At-Risk", "Average"])
    
    # Type 3: quiet achiever
    for i in chunks[2]:
        df.loc[i, "attendance"] = np.random.uniform(40, 58)
        df.loc[i, "previous_score"] = np.random.uniform(80, 95)
        df.loc[i, "study_hours"] = np.random.uniform(5, 7)
        labels[df.index.get_loc(i)] = np.random.choice(["Average", "Good"])
    
    # Type 4: pure label noise
    for i in chunks[3]:
        labels[df.index.get_loc(i)] = np.random.choice(
            ["At-Risk", "Average", "Good"]
        )
    
    return df, labels


def generate_dataset(n: int = N_STUDENTS, n_edge: int = N_EDGE_CASES) -> pd.DataFrame:
    """
    Generate the complete dataset with all features and target labels.
    """
    df = generate_base_features(n)
    latent = compute_latent_ability(df)
    labels = assign_performance_level(latent)
    df, labels = inject_edge_cases(df, labels, n_edge)
    df["performance_level"] = labels
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating student performance dataset...")
    dataset = generate_dataset()
    
    out_path = data_dir / "dataset.csv"
    dataset.to_csv(out_path, index=False)
    
    print(f"Dataset generated: {out_path} ({len(dataset)} rows)")
    print("\nClass distribution:")
    print(dataset["performance_level"].value_counts())
    print("\nClass distribution (%):")
    print(dataset["performance_level"].value_counts(normalize=True) * 100)
    print("\nSample rows (first 5):")
    print(dataset.head(5).to_string(index=False))
    
    print("\nFeature statistics:")
    print(dataset.describe()