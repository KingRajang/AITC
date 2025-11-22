#!/usr/bin/env python3
"""
AITC Project Verification Script
Tests all components after reorganization to ensure everything still works
"""

import sys
import os
from pathlib import Path
import json

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    print(f"  {text}")

class ProjectVerifier:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        
    def verify_directory_structure(self):
        """Verify that all essential directories exist"""
        print_header("1. Verifying Directory Structure")
        
        essential_dirs = [
            'config',
            'data',
            'data/input_images',
            'data/output_images',
            'models',
            'src',
            'src/vision',
            'src/simulation',
            'src/agent',
            'scripts',
            'notebooks',
            'results',
            'archive'
        ]
        
        for dir_path in essential_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print_success(f"Directory exists: {dir_path}/")
            else:
                print_error(f"Directory missing: {dir_path}/")
                self.errors.append(f"Missing directory: {dir_path}")
    
    def verify_config_files(self):
        """Verify configuration files exist and are valid JSON"""
        print_header("2. Verifying Configuration Files")
        
        config_files = [
            'config/lane_config.json',
            'config/lane_mapping.json',
            'config/sim_config.json',
            'data/initial_state.json'
        ]
        
        for config_file in config_files:
            full_path = self.project_root / config_file
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        json.load(f)
                    print_success(f"Config valid: {config_file}")
                except json.JSONDecodeError as e:
                    print_error(f"Invalid JSON in {config_file}: {e}")
                    self.errors.append(f"Invalid JSON: {config_file}")
            else:
                print_error(f"Config missing: {config_file}")
                self.errors.append(f"Missing config: {config_file}")
    
    def verify_source_files(self):
        """Verify all essential source code files exist"""
        print_header("3. Verifying Source Code Files")
        
        source_files = [
            'src/__init__.py',
            'src/vision/__init__.py',
            'src/vision/yolo_processor.py',
            'src/vision/dbscan_analyzer.py',
            'src/simulation/__init__.py',
            'src/simulation/environment.py',
            'src/simulation/reward_calculator.py',
            'src/simulation/traffic_models.py',
            'src/agent/__init__.py',
            'src/agent/q_learning_agent.py'
        ]
        
        for source_file in source_files:
            full_path = self.project_root / source_file
            if full_path.exists():
                print_success(f"Source exists: {source_file}")
            else:
                print_error(f"Source missing: {source_file}")
                self.errors.append(f"Missing source: {source_file}")
    
    def verify_script_files(self):
        """Verify essential script files exist"""
        print_header("4. Verifying Script Files")
        
        script_files = [
            'scripts/main_vision_processing.py',
            'scripts/main_rl_training.py',
            'scripts/ablation_1_full_system.py',
            'scripts/ablation_2_without_yolo.py',
            'scripts/ablation_3_without_dbscan.py',
            'scripts/ablation_4_without_rl.py',
            'scripts/actuated_baseline.py',
            'scripts/fixed_time_baseline.py',
            'scripts/evaluate_ql.py',
            'run_experiments.py'
        ]
        
        for script_file in script_files:
            full_path = self.project_root / script_file
            if full_path.exists():
                print_success(f"Script exists: {script_file}")
            else:
                print_warning(f"Script missing: {script_file}")
                self.warnings.append(f"Missing script: {script_file}")
    
    def verify_data_files(self):
        """Verify essential data files exist"""
        print_header("5. Verifying Data Files")
        
        data_files = [
            'data/input_images/lane_north.jpg',
            'data/input_images/lane_south.jpg',
            'data/input_images/lane_east.jpg',
            'data/input_images/lane_west.jpg'
        ]
        
        for data_file in data_files:
            full_path = self.project_root / data_file
            if full_path.exists():
                print_success(f"Data exists: {data_file}")
            else:
                print_error(f"Data missing: {data_file}")
                self.errors.append(f"Missing data: {data_file}")
    
    def verify_imports(self):
        """Verify that Python modules can be imported"""
        print_header("6. Verifying Python Imports")
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        modules = [
            ('src.vision.yolo_processor', 'YOLOProcessor'),
            ('src.vision.dbscan_analyzer', 'DBSCANAnalyzer'),
            ('src.simulation.environment', 'JammingMachine'),
            ('src.simulation.reward_calculator', 'RewardCalculator'),
            ('src.simulation.traffic_models', 'TrafficModel'),
            ('src.agent.q_learning_agent', 'QLearningAgent')
        ]
        
        for module_name, class_name in modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print_success(f"Import successful: {module_name}.{class_name}")
            except ImportError as e:
                print_error(f"Import failed: {module_name}.{class_name}")
                print_info(f"Error: {e}")
                self.errors.append(f"Import failed: {module_name}")
            except AttributeError as e:
                print_error(f"Class not found: {module_name}.{class_name}")
                print_info(f"Error: {e}")
                self.errors.append(f"Class not found: {class_name}")
    
    def verify_dependencies(self):
        """Verify that required packages are installed"""
        print_header("7. Verifying Dependencies")
        
        required_packages = [
            'numpy',
            'opencv-cv2',
            'scikit-learn',
            'ultralytics',
            'matplotlib',
            'pandas',
            'jupyter',
            'PIL'
        ]
        
        for package in required_packages:
            try:
                if package == 'opencv-cv2':
                    __import__('cv2')
                    print_success(f"Package installed: opencv-python")
                elif package == 'PIL':
                    __import__('PIL')
                    print_success(f"Package installed: Pillow")
                else:
                    __import__(package)
                    print_success(f"Package installed: {package}")
            except ImportError:
                print_warning(f"Package missing: {package}")
                self.warnings.append(f"Missing package: {package}")
    
    def verify_notebooks(self):
        """Verify notebook files exist"""
        print_header("8. Verifying Notebooks")
        
        notebooks = [
            'notebooks/1_vision_pipeline_testbed.ipynb',
            'notebooks/2_results_analysis.ipynb',
            'notebooks/3_enhanced_baseline_comparison.ipynb'
        ]
        
        for notebook in notebooks:
            full_path = self.project_root / notebook
            if full_path.exists():
                print_success(f"Notebook exists: {notebook}")
            else:
                print_warning(f"Notebook missing: {notebook}")
                self.warnings.append(f"Missing notebook: {notebook}")
    
    def verify_archive_structure(self):
        """Verify archive structure is organized"""
        print_header("9. Verifying Archive Structure")
        
        archive_dirs = [
            'archive/backup',
            'archive/debug_files',
            'archive/old_experiments'
        ]
        
        for archive_dir in archive_dirs:
            full_path = self.project_root / archive_dir
            if full_path.exists():
                print_success(f"Archive exists: {archive_dir}/")
            else:
                print_info(f"Archive directory not found (OK if not needed): {archive_dir}/")
    
    def verify_cleaned_files(self):
        """Verify that cleaned files are actually removed"""
        print_header("10. Verifying Cleanup")
        
        should_be_removed = [
            'check_state_coverage_FIXED.py',
            'fix_imports.py',
            'reorganize_project.py',
            'setup_ablation.sh',
            'setup_experiments.py',
            'structure.txt',
            'trained_q_table.json',
            'trained_q_table.npy',
            'training_log.json'
        ]
        
        for file_path in should_be_removed:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print_success(f"Cleaned: {file_path}")
            else:
                print_warning(f"Still exists: {file_path}")
                self.warnings.append(f"File should be removed: {file_path}")
    
    def generate_report(self):
        """Generate final verification report"""
        print_header("Verification Report")
        
        if not self.errors and not self.warnings:
            print_success("ALL CHECKS PASSED! ✨")
            print_info("Your project structure is clean and all components are accessible.")
            return True
        
        if self.errors:
            print_error(f"Found {len(self.errors)} critical errors:")
            for error in self.errors:
                print_info(f"  • {error}")
            print()
        
        if self.warnings:
            print_warning(f"Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print_info(f"  • {warning}")
            print()
        
        if self.errors:
            print_error("⚠️  CRITICAL ERRORS FOUND - Please fix before running experiments")
            return False
        else:
            print_success("✓ No critical errors - Project should work (warnings are minor)")
            return True

def main():
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         AITC PROJECT VERIFICATION SCRIPT                   ║")
    print("║  Testing all components after reorganization               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    # Determine project root
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path.cwd()
    
    print_info(f"Project root: {project_root}\n")
    
    # Run verification
    verifier = ProjectVerifier(project_root)
    
    verifier.verify_directory_structure()
    verifier.verify_config_files()
    verifier.verify_source_files()
    verifier.verify_script_files()
    verifier.verify_data_files()
    verifier.verify_imports()
    verifier.verify_dependencies()
    verifier.verify_notebooks()
    verifier.verify_archive_structure()
    verifier.verify_cleaned_files()
    
    success = verifier.generate_report()
    
    if success:
        print(f"\n{GREEN}🎉 Ready to run experiments!{RESET}")
        print_info("Try running: python run_experiments.py")
    else:
        print(f"\n{RED}❌ Please fix critical errors before proceeding{RESET}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()