#!/bin/bash

# AITC Project Cleanup Script
# This script safely reorganizes the project structure with backups

echo "🚦 AITC Project Cleanup Script"
echo "================================"
echo ""

# Get the project root directory
PROJECT_ROOT="$(pwd)"

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/readme.md" ]; then
    echo "❌ Error: Cannot find AITC project at $PROJECT_ROOT"
    echo "Please update PROJECT_ROOT in this script to match your project location"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "✅ Found project at: $PROJECT_ROOT"
echo ""

# Create backup before making changes
BACKUP_DIR="archive/cleanup_backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup files we're about to move/delete
echo "   Backing up files..."
[ -f "trained_q_table.json" ] && cp trained_q_table.json "$BACKUP_DIR/"
[ -f "trained_q_table.npy" ] && cp trained_q_table.npy "$BACKUP_DIR/"
[ -f "training_log.json" ] && cp training_log.json "$BACKUP_DIR/"
[ -f "check_state_coverage_FIXED.py" ] && cp check_state_coverage_FIXED.py "$BACKUP_DIR/"
[ -f "fix_imports.py" ] && cp fix_imports.py "$BACKUP_DIR/"
[ -f "reorganize_project.py" ] && cp reorganize_project.py "$BACKUP_DIR/"
[ -f "setup_ablation.sh" ] && cp setup_ablation.sh "$BACKUP_DIR/"
[ -f "setup_experiments.py" ] && cp setup_experiments.py "$BACKUP_DIR/"
[ -f "structure.txt" ] && cp structure.txt "$BACKUP_DIR/"

echo "✅ Backup created successfully"
echo ""

# Step 1: Move old training files from root
echo "📁 Step 1: Moving old training files..."
mkdir -p archive/old_training_root
[ -f "trained_q_table.json" ] && mv trained_q_table.json archive/old_training_root/ && echo "   ✓ Moved trained_q_table.json"
[ -f "trained_q_table.npy" ] && mv trained_q_table.npy archive/old_training_root/ && echo "   ✓ Moved trained_q_table.npy"
[ -f "training_log.json" ] && mv training_log.json archive/old_training_root/ && echo "   ✓ Moved training_log.json"
echo ""

# Step 2: Remove Zone.Identifier files
echo "🗑️  Step 2: Removing Zone.Identifier files..."
ZONE_COUNT=$(find data/input_images -name "*:Zone.Identifier" 2>/dev/null | wc -l)
if [ $ZONE_COUNT -gt 0 ]; then
    find data/input_images -name "*:Zone.Identifier" -delete
    echo "   ✓ Removed $ZONE_COUNT Zone.Identifier files"
else
    echo "   ✓ No Zone.Identifier files found"
fi
echo ""

# Step 3: Remove temporary/utility files
echo "🗑️  Step 3: Removing temporary utility files..."
[ -f "check_state_coverage_FIXED.py" ] && rm check_state_coverage_FIXED.py && echo "   ✓ Removed check_state_coverage_FIXED.py"
[ -f "fix_imports.py" ] && rm fix_imports.py && echo "   ✓ Removed fix_imports.py"
[ -f "reorganize_project.py" ] && rm reorganize_project.py && echo "   ✓ Removed reorganize_project.py"
[ -f "setup_ablation.sh" ] && rm setup_ablation.sh && echo "   ✓ Removed setup_ablation.sh"
[ -f "setup_experiments.py" ] && rm setup_experiments.py && echo "   ✓ Removed setup_experiments.py"
[ -f "structure.txt" ] && rm structure.txt && echo "   ✓ Removed structure.txt"
echo ""

# Step 4: Move diagnostic scripts to archive
echo "📁 Step 4: Moving diagnostic scripts to archive..."
[ -f "scripts/check_rl_timing.py" ] && mv scripts/check_rl_timing.py archive/debug_files/ && echo "   ✓ Moved check_rl_timing.py"
[ -f "scripts/final_comprehensive_test.py" ] && mv scripts/final_comprehensive_test.py archive/debug_files/ && echo "   ✓ Moved final_comprehensive_test.py"
echo ""

# Step 5: Clean empty directories
echo "🧹 Step 5: Cleaning empty directories..."
find results -type d -empty -delete 2>/dev/null && echo "   ✓ Cleaned empty directories in results/"
find docs -type d -empty -delete 2>/dev/null && echo "   ✓ Cleaned empty directories in docs/"
echo ""

# Step 6: Generate new structure file
echo "📄 Step 6: Generating clean structure file..."
if command -v tree &> /dev/null; then
    tree -I 'venv|__pycache__|*.pyc|.git' > project_structure_clean.txt
    echo "   ✓ Created project_structure_clean.txt"
else
    find . -not -path '*/\.*' -not -path '*/venv/*' -not -path '*/__pycache__/*' | sort > project_structure_clean.txt
    echo "   ✓ Created project_structure_clean.txt (using find)"
fi
echo ""

echo "✅ Cleanup completed successfully!"
echo ""
echo "📊 Summary:"
echo "   • Backup created at: $BACKUP_DIR"
echo "   • Old training files moved to: archive/old_training_root/"
echo "   • Diagnostic scripts moved to: archive/debug_files/"
echo "   • Temporary files removed"
echo "   • Zone.Identifier files removed"
echo ""
echo "🔍 Next step: Run the verification script to test everything works!"
echo ""