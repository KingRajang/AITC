import subprocess, sys
scripts = ["scripts/ablation_1_full_system.py", "scripts/ablation_2_without_yolo.py",
           "scripts/ablation_3_without_dbscan.py", "scripts/ablation_4_without_rl.py"]
print("🚀 Running Ablation Study (400 episodes)...\n")
for s in scripts:
    print(f"Running {s}...")
    subprocess.run([sys.executable, s])
print("\n✅ Done! Now run: python3 scripts/analyze_ablation.py")
