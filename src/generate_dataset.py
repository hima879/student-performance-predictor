# generate_dataset.py

import numpy as np
import pandas as pd
import os

# This makes sure we all get the same random numbers
np.random.seed(42)

# Number of students
N = 1000

print("=" * 50)
print("📊 GENERATING STUDENT DATASET - ENHANCED")
print("=" * 50)
print(f"   Creating {N} student records...")

# ========== GENERATE BASE FEATURES ==========

# 1. Study Hours (with some randomness)
study_hours = np.clip(np.random.normal(5, 2, N), 0, 12)
# Add realistic noise
study_hours += np.random.normal(0, 0.3, N)
study_hours = np.clip(study_hours, 0, 12)

# 2. Attendance (with some randomness)
attendance = np.clip(np.random.normal(75, 15, N), 30, 100)
# Add realistic noise
attendance += np.random.normal(0, 2, N)
attendance = np.clip(attendance, 30, 100)

# 3. Previous Score
previous_score = np.clip(np.random.normal(65, 15, N), 20, 100).astype(int)

# 4. Assignments Completed
assignments = np.clip(np.random.normal(10, 4, N), 0, 20).astype(int)

# 5. Sleep Hours
sleep_hours = np.clip(np.random.normal(7, 1.5, N), 3, 10)

# 6. Categorical Features
parental_support = np.random.choice(['Low','Medium','High'], N, p=[0.25, 0.45, 0.30])
extra_curricular = np.random.choice(['Yes','No'], N, p=[0.55, 0.45])
internet_access = np.random.choice(['Yes','No'], N, p=[0.80, 0.20])

# ========== ADD "SURPRISING" CASES ==========
print("   🔄 Adding realistic edge cases...")

# Case 1: Bright student who does poorly (20 students)
bright_struggling = np.random.choice(N, 20, replace=False)
study_hours[bright_struggling] = np.random.uniform(2, 4, 20)  # Low study
attendance[bright_struggling] = np.random.uniform(40, 60, 20)  # Low attendance
# But high previous score
previous_score[bright_struggling] = np.random.randint(80, 95, 20)

# Case 2: Struggling student who performs well (20 students)
struggling_brilliant = np.random.choice(N, 20, replace=False)
# Avoid overlap with first group
struggling_brilliant = [i for i in struggling_brilliant if i not in bright_struggling][:20]
study_hours[struggling_brilliant] = np.random.uniform(8, 11, 20)  # High study
attendance[struggling_brilliant] = np.random.uniform(85, 100, 20)  # High attendance
previous_score[struggling_brilliant] = np.random.randint(30, 50, 20)  # Low previous score

# Case 3: No sleep but high performance (10 students)
no_sleep_high_performer = np.random.choice(N, 10, replace=False)
sleep_hours[no_sleep_high_performer] = np.random.uniform(3, 5, 10)
# But everything else is good
study_hours[no_sleep_high_performer] = np.random.uniform(7, 10, 10)
attendance[no_sleep_high_performer] = np.random.uniform(85, 100, 10)
assignments[no_sleep_high_performer] = np.random.randint(15, 20, 10)

print(f"   ✅ Added {20 + 20 + 10} realistic edge cases")

# ========== CALCULATE PERFORMANCE SCORE ==========

# Convert categorical to numeric scores
support_score = np.where(parental_support=='High', 1, 
                         np.where(parental_support=='Medium', 0.5, 0))
ec_score = np.where(extra_curricular=='Yes', 0.5, 0)
net_score = np.where(internet_access=='Yes', 0.3, 0)

# Weighted formula (now with more realistic weights)
weighted = (
    # Academic factors (70% of weight)
    study_hours * 3.0 +           # Study hours
    (attendance / 100) * 15.0 +   # Attendance
    (previous_score / 100) * 25.0 + # Previous score
    (assignments / 20) * 10.0 +   # Assignments
    sleep_hours * 1.0 +           # Sleep (reduced weight)
    
    # Support factors (20% of weight)
    support_score * 8.0 +         # Parental support
    ec_score * 3.0 +              # Extra curricular
    net_score * 2.0 +             # Internet access
    
    # Random noise (10% of weight)
    np.random.normal(0, 5, N)     # More noise for realism
)

# Add some non-linear effects
# Students with very high study hours but poor sleep get penalized
high_study_low_sleep = (study_hours > 8) & (sleep_hours < 5)
weighted[high_study_low_sleep] -= 5

# Students with good attendance and good study get bonus
good_pattern = (attendance > 85) & (study_hours > 6)
weighted[good_pattern] += 3

# ========== CREATE PERFORMANCE CATEGORIES ==========

# Split into 3 equal groups
p33, p66 = np.percentile(weighted, 33), np.percentile(weighted, 66)
performance = np.where(weighted >= p66, 'Good', 
                       np.where(weighted >= p33, 'Average', 'At-Risk'))

# Manually override some cases to make it more realistic
# Some "bright but struggling" students become Average instead of At-Risk
for idx in bright_struggling[:10]:
    if performance[idx] == 'At-Risk':
        # 50% chance they're Average instead
        if np.random.random() > 0.5:
            performance[idx] = 'Average'

# Some "struggling but brilliant" students become Good instead of Average
for idx in struggling_brilliant[:10]:
    if performance[idx] == 'Average':
        # 50% chance they're Good instead
        if np.random.random() > 0.5:
            performance[idx] = 'Good'

# ========== CREATE FINAL DATAFRAME ==========

df = pd.DataFrame({
    'student_id': range(1001, 1001 + N),
    'study_hours_per_day': study_hours.round(1),
    'attendance_percentage': attendance.round(1),
    'previous_score': previous_score,
    'assignments_completed': assignments,
    'sleep_hours': sleep_hours.round(1),
    'parental_support': parental_support,
    'extra_curricular': extra_curricular,
    'internet_access': internet_access,
    'performance_level': performance
})

# ========== SAVE TO CSV ==========

# Get the current directory
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'dataset.csv')

# Save to CSV
df.to_csv(file_path, index=False)

# ========== DISPLAY STATISTICS ==========

print(f"\n✅ Dataset created successfully!")
print(f"   📁 Saved to: {file_path}")
print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📊 Performance breakdown:")
print(df['performance_level'].value_counts())
print(f"\n📊 Performance percentages:")
print((df['performance_level'].value_counts() / N * 100).round(1))

print(f"\n📊 Feature Statistics:")
print(f"   Study Hours: {df['study_hours_per_day'].mean():.1f} ± {df['study_hours_per_day'].std():.1f}")
print(f"   Attendance: {df['attendance_percentage'].mean():.1f} ± {df['attendance_percentage'].std():.1f}")
print(f"   Previous Score: {df['previous_score'].mean():.1f} ± {df['previous_score'].std():.1f}")
print(f"   Assignments: {df['assignments_completed'].mean():.1f} ± {df['assignments_completed'].std():.1f}")
print(f"   Sleep: {df['sleep_hours'].mean():.1f} ± {df['sleep_hours'].std():.1f}")

print(f"\n📋 First 5 rows:")
print(df.head())

print("\n" + "=" * 50)
print("✅ DATASET GENERATION COMPLETE!")
print("=" * 50)
print("\n🚀 Next step: Run 'python model.py'")
# Save to CSV
df.to_csv(file_path, index=False)

