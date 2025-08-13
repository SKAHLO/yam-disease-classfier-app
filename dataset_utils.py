#!/usr/bin/env python3
"""
Dataset utilities for yam leaf disease classification
Includes data augmentation, validation, and preprocessing tools
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import random
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, dataset_path: str = "yam_disease_dataset"):
        self.dataset_path = Path(dataset_path)
        self.categories = [
            'healthy', 'anthracnose', 'leaf_spot', 'leaf_blight',
            'mosaic_virus', 'mild_mosaic', 'bacterial_spot', 'bacilliform_virus'
        ]
        
    def validate_dataset(self) -> Dict:
        """Validate the scraped dataset"""
        results = {}
        total_images = 0
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                results[category] = {'count': 0, 'valid': 0, 'invalid': []}
                continue
                
            images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            valid_count = 0
            invalid_files = []
            
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's a valid image
                    valid_count += 1
                except Exception as e:
                    invalid_files.append(str(img_path))
                    logger.warning(f"Invalid image: {img_path} - {e}")
            
            results[category] = {
                'count': len(images),
                'valid': valid_count,
                'invalid': invalid_files
            }
            total_images += valid_count
        
        results['total_valid'] = total_images
        logger.info(f"Dataset validation complete. Total valid images: {total_images}")
        return results
    
    def augment_images(self, target_count: int = 200):
        """Augment images to balance the dataset"""
        logger.info("Starting data augmentation...")
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            current_count = len(images)
            
            if current_count >= target_count:
                logger.info(f"{category}: {current_count} images (sufficient)")
                continue
                
            needed = target_count - current_count
            logger.info(f"{category}: {current_count} images, need {needed} more")
            
            augmented_count = 0
            while augmented_count < needed and images:
                source_img = random.choice(images)
                
                try:
                    augmented_img = self.apply_augmentation(source_img)
                    if augmented_img:
                        # Save augmented image
                        aug_filename = f"aug_{augmented_count}_{source_img.stem}.jpg"
                        aug_path = category_path / aug_filename
                        augmented_img.save(aug_path, "JPEG", quality=85)
                        augmented_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to augment {source_img}: {e}")
                    
        logger.info("Data augmentation complete")
    
    def apply_augmentation(self, image_path: Path) -> Image.Image:
        """Apply random augmentations to an image"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Random augmentations
                augmentations = [
                    lambda x: x.rotate(random.randint(-15, 15), expand=True),
                    lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
                    lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
                    lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
                    lambda x: ImageOps.mirror(x) if random.random() > 0.5 else x,
                    lambda x: ImageOps.flip(x) if random.random() > 0.8 else x,
                ]
                
                # Apply 1-3 random augmentations
                num_augs = random.randint(1, 3)
                selected_augs = random.sample(augmentations, num_augs)
                
                for aug in selected_augs:
                    img = aug(img)
                    
                return img
                
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            return None
    
    def create_train_val_split(self, train_ratio: float = 0.8):
        """Split dataset into training and validation sets"""
        logger.info(f"Creating train/val split ({train_ratio:.0%} train)")
        
        # Create split directories
        splits = ['train', 'val']
        for split in splits:
            split_path = self.dataset_path / split
            split_path.mkdir(exist_ok=True)
            
            for category in self.categories:
                (split_path / category).mkdir(exist_ok=True)
        
        # Split each category
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            random.shuffle(images)
            
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Move images to respective splits
            for img in train_images:
                dest = self.dataset_path / 'train' / category / img.name
                shutil.copy2(img, dest)
                
            for img in val_images:
                dest = self.dataset_path / 'val' / category / img.name
                shutil.copy2(img, dest)
                
            logger.info(f"{category}: {len(train_images)} train, {len(val_images)} val")
    
    def resize_images(self, target_size: Tuple[int, int] = (224, 224)):
        """Resize all images to target size for model training"""
        logger.info(f"Resizing images to {target_size}")
        
        # Process all splits and categories
        for split in ['train', 'val']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            for category in self.categories:
                category_path = split_path / category
                if not category_path.exists():
                    continue
                    
                images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
                
                for img_path in images:
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            img = img.resize(target_size, Image.Resampling.LANCZOS)
                            img.save(img_path, "JPEG", quality=85)
                            
                    except Exception as e:
                        logger.error(f"Failed to resize {img_path}: {e}")
    
    def generate_labels_file(self):
        """Generate labels.txt file for the dataset"""
        labels_path = self.dataset_path / 'labels.txt'
        
        with open(labels_path, 'w') as f:
            for i, category in enumerate(self.categories):
                f.write(f"{i}: {category}\n")
                
        logger.info(f"Labels file created: {labels_path}")
        
        # Also create class mapping JSON
        class_mapping = {i: category for i, category in enumerate(self.categories)}
        mapping_path = self.dataset_path / 'class_mapping.json'
        
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
            
        logger.info(f"Class mapping created: {mapping_path}")
    
    def cleanup_dataset(self):
        """Remove invalid images and organize dataset"""
        logger.info("Cleaning up dataset...")
        
        removed_count = 0
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            for img_path in category_path.iterdir():
                if not img_path.is_file():
                    continue
                    
                try:
                    with Image.open(img_path) as img:
                        # Check image properties
                        if img.size[0] < 50 or img.size[1] < 50:  # Too small
                            img_path.unlink()
                            removed_count += 1
                            continue
                            
                        img.verify()
                        
                except Exception:
                    img_path.unlink()
                    removed_count += 1
                    
        logger.info(f"Removed {removed_count} invalid images")

def main():
    """Process the scraped dataset"""
    print("Yam Disease Dataset Processor")
    print("="*40)
    
    processor = DatasetProcessor()
    
    # Validate dataset
    validation_results = processor.validate_dataset()
    print(f"Valid images found: {validation_results['total_valid']}")
    
    # Clean up dataset
    processor.cleanup_dataset()
    
    # Augment to balance classes
    processor.augment_images(target_count=150)
    
    # Create train/validation split
    processor.create_train_val_split(train_ratio=0.8)
    
    # Resize images for training
    processor.resize_images(target_size=(224, 224))
    
    # Generate labels
    processor.generate_labels_file()
    
    print("\nDataset processing complete!")
    print("Your dataset is ready for machine learning training.")
    print("\nDataset structure:")
    print("├── train/")
    print("│   ├── healthy/")
    print("│   ├── anthracnose/")
    print("│   └── ...")
    print("├── val/")
    print("│   ├── healthy/")
    print("│   └── ...")
    print("├── labels.txt")
    print("└── class_mapping.json")

if __name__ == "__main__":
    main()
