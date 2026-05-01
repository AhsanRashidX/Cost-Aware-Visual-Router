"""
Manual download of sentence transformer model with error handling
"""

import os
import sys
import requests
from pathlib import Path
import time

print("="*60)
print("Manual Download of Sentence Transformer Model")
print("="*60)

# Create directories
cache_dir = Path("./cache/sentence_transformers")
cache_dir.mkdir(parents=True, exist_ok=True)

models_dir = Path("./models/all-mpnet-base-v2")
models_dir.mkdir(parents=True, exist_ok=True)

print(f"\nCache directory: {cache_dir.absolute()}")
print(f"Models directory: {models_dir.absolute()}")

# Try downloading with huggingface_hub
print("\n[1/3] Attempting download using huggingface_hub...")

try:
    from huggingface_hub import snapshot_download
    
    print("Downloading all-mpnet-base-v2 model...")
    snapshot_download(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✅ Download complete via huggingface_hub!")
    
except ImportError:
    print("huggingface_hub not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
    
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False
    )
    print("✅ Download complete!")

except Exception as e:
    print(f"❌ huggingface_hub download failed: {e}")
    
    # Option 2: Try direct download of key files
    print("\n[2/3] Trying direct file downloads...")
    
    files_to_download = {
        "pytorch_model.bin": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/pytorch_model.bin",
        "config.json": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json",
        "tokenizer_config.json": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer_config.json",
        "vocab.txt": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/vocab.txt",
        "special_tokens_map.json": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/special_tokens_map.json",
        "modules.json": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/modules.json",
    }
    
    for filename, url in files_to_download.items():
        filepath = models_dir / filename
        if filepath.exists():
            print(f"   {filename} already exists, skipping...")
            continue
        
        print(f"   Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✅ Downloaded {filename} ({filepath.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"   ❌ Failed to download {filename}: {e}")
    
    # Download model.safetensors if available
    print("\n[3/3] Trying safetensors format...")
    safetensors_url = "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/model.safetensors"
    safetensors_path = models_dir / "model.safetensors"
    
    try:
        response = requests.get(safetensors_url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(safetensors_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✅ Downloaded model.safetensors ({safetensors_path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print("   ℹ️ model.safetensors not available, using pytorch_model.bin")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# Verify download
print("\n" + "="*60)
print("Verifying Download")
print("="*60)

required_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
missing_files = []

for f in required_files:
    if (models_dir / f).exists():
        size = (models_dir / f).stat().st_size / 1024 / 1024
        print(f"   ✅ {f} ({size:.2f} MB)")
    else:
        print(f"   ❌ {f} missing")
        missing_files.append(f)

if not missing_files:
    print("\n✅ Model downloaded successfully!")
    
    # Test loading
    print("\nTesting model loading...")
    try:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(models_dir))
        embedding = model.encode("test query")
        print(f"✅ Model loads successfully! Embedding shape: {embedding.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
else:
    print(f"\n❌ Missing files: {missing_files}")
    print("\nTry running this command instead:")
    print("   huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir ./models/all-mpnet-base-v2")