#!/usr/bin/env python3
"""Project Reorganization Script"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("🗂️  PROJECT REORGANIZATION SCRIPT")
    print("=" * 70)
    
    project_root = Path.cwd()
    print(f"\nProject root: {project_root}")
    
    response = input("\nContinue? (yes/no): ").lower().strip()
    if response != 'yes':
        print("❌ Cancelled.")
        return
    
    # Create directories
    directories = [
        "models", "scripts", "docs",
        "archive/debug_files", "archive/old_experiments", "archive/backup",
        "results/figures", "results/tables", "results/logs",
    ]
    
    print("\n📁 Creating directories...")
    for d in directories:
        (project_root / d).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d}")
    
    # Move model files
    print("\n📦 Moving models...")
    if (project_root / "trained_q_table.json").exists():
        dst = project_root / "models" / "trained_q_table.json"
        if not dst.exists():
            shutil.move(str(project_root / "trained_q_table.json"), str(dst))
            print("   ✓ trained_q_table.json → models/")
    
    # Move scripts
    scripts = [
        "main_rl_training.py", "fresh_baseline_eval.py",
        "final_comprehensive_test.py", "check_rl_timing.py",
        "main_vision_processing.py"
    ]
    
    print("\n📜 Moving scripts...")
    for s in scripts:
        src = project_root / s
        dst = project_root / "scripts" / s
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"   ✓ {s} → scripts/")
    
    # Archive debug files
    debug = [
        "analyze_q_table_actions.py", "check_rewards.py",
        "check_state_coverage.py", "check_zero_zero_state.py",
        "diagnose_evaluation_states.py", "diagnostic.py",
        "quick_analysis.py", "test_baselines_simple.py",
        "verify_qtable.py", "test_with_random_fallback.py"
    ]
    
    print("\n📦 Archiving debug files...")
    for d in debug:
        src = project_root / d
        if src.exists():
            dst = project_root / "archive" / "debug_files" / d
            shutil.move(str(src), str(dst))
            print(f"   ✓ {d} → archive/")
    
    # Archive old files
    if (project_root / "trained_q_table_OLD.json").exists():
        shutil.move(
            str(project_root / "trained_q_table_OLD.json"),
            str(project_root / "archive" / "old_experiments" / "trained_q_table_OLD.json")
        )
        print("\n✓ trained_q_table_OLD.json → archive/")
    
    # Move docs
    docs = ["COMPLETE_PROJECT_SUMMARY.md", "THESIS_WRITING_GUIDE.md", 
            "QUICK_REFERENCE.md", "PROJECT_REORGANIZATION_GUIDE.md"]
    
    print("\n📄 Moving docs...")
    for doc in docs:
        src = project_root / doc
        if src.exists():
            dst = project_root / "docs" / doc
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"   ✓ {doc} → docs/")
    
    print("\n" + "=" * 70)
    print("✅ REORGANIZATION COMPLETE!")
    print("=" * 70)
    print("\n💡 Test scripts: python scripts/main_rl_training.py")

if __name__ == "__main__":
    main()
