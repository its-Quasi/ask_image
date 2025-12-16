"""
Verification script to check if all required models are properly installed.
Run this script before starting the application to ensure everything is set up correctly.
"""

from pathlib import Path
import sys


def verify_models():
    """Verify that all required model files are present."""

    models_dir = Path("models")

    # Required files with their approximate expected sizes (in MB)
    required_files = {
        "groundingdino_swint_ogc.pth": (600, 800),  # Expected range: 600-800 MB
        "GroundingDINO_SwinT_OGC.py": (0.001, 0.01),  # Expected: ~5 KB
        "sam2_hiera_large.pt": (800, 1000),  # Expected range: 800-1000 MB
    }

    # Alternative SAM2 models (if using different variant)
    alternative_sam2 = [
        "sam2_hiera_base_plus.pt",
        "sam2_hiera_small.pt",
        "sam2_hiera_tiny.pt",
    ]

    print("=" * 70)
    print("MODEL VERIFICATION")
    print("=" * 70)
    print()

    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        print(f"  Create it with: mkdir -p {models_dir}")
        return False

    print(f"Models directory: {models_dir.absolute()}")
    print("-" * 70)

    all_present = True
    total_size = 0

    # Check Grounding DINO files
    print("\n[Grounding DINO Models]")
    for filename in ["groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.py"]:
        filepath = models_dir / filename
        min_size, max_size = required_files[filename]

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb

            if min_size <= size_mb <= max_size:
                print(f"  ✓ {filename:<35} ({size_mb:>7.1f} MB) [OK]")
            else:
                print(f"  ⚠ {filename:<35} ({size_mb:>7.1f} MB) [Size Warning]")
                print(f"    Expected: {min_size}-{max_size} MB")
        else:
            print(f"  ✗ {filename:<35} [NOT FOUND]")
            all_present = False

    # Check SAM2 files
    print("\n[SAM2 Models]")
    sam2_found = False

    # Check for the default large model
    sam2_large = models_dir / "sam2_hiera_large.pt"
    if sam2_large.exists():
        size_mb = sam2_large.stat().st_size / (1024 * 1024)
        total_size += size_mb
        min_size, max_size = required_files["sam2_hiera_large.pt"]

        if min_size <= size_mb <= max_size:
            print(f"  ✓ sam2_hiera_large.pt                ({size_mb:>7.1f} MB) [OK]")
        else:
            print(f"  ⚠ sam2_hiera_large.pt                ({size_mb:>7.1f} MB) [Size Warning]")
        sam2_found = True

    # Check for alternative SAM2 models
    for alt_model in alternative_sam2:
        alt_path = models_dir / alt_model
        if alt_path.exists():
            size_mb = alt_path.stat().st_size / (1024 * 1024)
            if not sam2_found:
                total_size += size_mb
            print(f"  ℹ {alt_model:<35} ({size_mb:>7.1f} MB) [Alternative]")
            sam2_found = True

    if not sam2_found:
        print(f"  ✗ No SAM2 model found")
        print(f"    Expected: sam2_hiera_large.pt or alternatives")
        all_present = False

    # Summary
    print()
    print("-" * 70)
    print(f"Total size: {total_size:.1f} MB")
    print("-" * 70)
    print()

    if all_present:
        print("✓ All required models are present!")
        print()
        print("You can now run the application:")
        print("  python app.py")
        return True
    else:
        print("✗ Some models are missing!")
        print()
        print("To download missing models:")
        print("  1. Run: ./download_models.sh")
        print("  2. Or see: MODELS_SETUP.md for manual instructions")
        return False


def verify_dependencies():
    """Verify that required Python packages are installed."""

    print()
    print("=" * 70)
    print("DEPENDENCY VERIFICATION")
    print("=" * 70)
    print()

    required_packages = [
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "supervision",
        "flask",
        "groundingdino",
        "sam2",
    ]

    all_installed = True

    for package in required_packages:
        try:
            if package == "cv2":
                __import__("cv2")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False

    print()
    print("-" * 70)

    if all_installed:
        print("✓ All required packages are installed!")
    else:
        print("✗ Some packages are missing!")
        print()
        print("To install missing packages:")
        print("  ./install_dependencies.sh")
        print("  or")
        print("  pip install -r requirements.txt")

    return all_installed


def main():
    """Main verification function."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Grounding DINO + SAM2 - Setup Verification" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Verify models
    models_ok = verify_models()

    # Verify dependencies
    deps_ok = verify_dependencies()

    # Final summary
    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print()

    if models_ok and deps_ok:
        print("✓✓✓ System is ready! You can start the application.")
        print()
        print("To start:")
        print("  python app.py")
        print()
        print("Then open: http://localhost:5000")
        return 0
    else:
        print("✗✗✗ Setup is incomplete. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
