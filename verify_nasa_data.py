# verify_nasa_data.py
import os

def verify_nasa_dataset():
    """Verify NASA dataset files are correctly placed"""
    
    nasa_path = 'data/nasa/'
    required_files = []
    
    # Generate required file names
    for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
        required_files.extend([
            f'train_{dataset}.txt',
            f'test_{dataset}.txt', 
            f'RUL_{dataset}.txt'
        ])
    
    print("Checking NASA dataset files...")
    print("=" * 50)
    
    missing_files = []
    found_files = []
    
    for file in required_files:
        file_path = os.path.join(nasa_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            found_files.append(f"✓ {file} ({file_size:.1f} KB)")
        else:
            missing_files.append(f"✗ {file}")
    
    # Display results
    if found_files:
        print("Found files:")
        for file in found_files:
            print(f"  {file}")
    
    if missing_files:
        print(f"\nMissing files:")
        for file in missing_files:
            print(f"  {file}")
        return False
    else:
        print(f"\n🎉 All NASA dataset files found! ({len(found_files)} files)")
        return True

if __name__ == "__main__":
    verify_nasa_dataset()