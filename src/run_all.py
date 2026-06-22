# run_all.py
# Run this once to set everything up

import subprocess
import sys
import os

print("🚀 Setting up Student Performance Predictor...")
print("=" * 50)

# Step 1: Generate dataset
print("\n📊 Step 1: Generating dataset...")
subprocess.run([sys.executable, "generate_dataset.py"])

# Step 2: Train model
print("\n🌲 Step 2: Training model...")
subprocess.run([sys.executable, "model.py"])

print("\n" + "=" * 50)
print("✅ Setup complete!")
print("\n🚀 To run the app:")
print("   streamlit run app.py")