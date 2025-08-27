#!/usr/bin/env python3
"""
Teachable Machine Model Downloader
Downloads your trained model from Teachable Machine for integration
"""

import requests
import zipfile
import os
import shutil
from pathlib import Path

# Your Teachable Machine model ID
MODEL_ID = "99_E2CS7s"
MODEL_URL = f"https://teachablemachine.withgoogle.com/models/{MODEL_ID}/"

def download_teachable_machine_model():
    """
    Downloads the TensorFlow Lite model from Teachable Machine
    """
    print(f"Downloading Teachable Machine model: {MODEL_ID}")
    
    # Create assets/models directory if it doesn't exist
    assets_dir = Path("assets/models")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the TensorFlow Lite model
        tflite_url = f"https://teachablemachine.withgoogle.com/models/{MODEL_ID}/model.tflite"
        labels_url = f"https://teachablemachine.withgoogle.com/models/{MODEL_ID}/labels.txt"
        
        print("Downloading model.tflite...")
        response = requests.get(tflite_url)
        if response.status_code == 200:
            with open("assets/models/yam_disease_model.tflite", "wb") as f:
                f.write(response.content)
            print("✅ Model file downloaded successfully!")
        else:
            print(f"❌ Failed to download model file: {response.status_code}")
            return False
        
        print("Downloading labels.txt...")
        response = requests.get(labels_url)
        if response.status_code == 200:
            with open("assets/models/labels.txt", "w") as f:
                f.write(response.text)
            print("✅ Labels file downloaded successfully!")
        else:
            print(f"❌ Failed to download labels file: {response.status_code}")
            return False
        
        print("\n🎉 Teachable Machine model integration complete!")
        print("Your app will now use the real trained model instead of intelligent analysis.")
        print("\nNext steps:")
        print("1. Run: flutter clean")
        print("2. Run: flutter pub get")
        print("3. Run: flutter run")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def manual_instructions():
    """
    Provides manual download instructions if automatic download fails
    """
    print("\n📋 Manual Download Instructions:")
    print("="*50)
    print(f"1. Go to: {MODEL_URL}")
    print("2. Click 'Export Model'")
    print("3. Select 'TensorFlow Lite' tab")
    print("4. Choose 'Floating point'")
    print("5. Click 'Download my model'")
    print("6. Extract the downloaded zip file")
    print("7. Copy files to your project:")
    print("   - Rename 'model.tflite' to 'yam_disease_model.tflite'")
    print("   - Place both files in: assets/models/")
    print("\nFiles needed:")
    print("   assets/models/yam_disease_model.tflite")
    print("   assets/models/labels.txt")

if __name__ == "__main__":
    print("Teachable Machine Model Downloader")
    print("="*40)
    
    if not download_teachable_machine_model():
        manual_instructions()
