# Yam Leaf Disease Dataset Scraper

This tool creates a comprehensive dataset of yam leaf disease images for machine learning and object detection models.

## Disease Categories

The scraper collects images for 8 categories:

1. **Healthy** - Normal, disease-free yam leaves
2. **Anthracnose** - Dark spots with yellow halos (Colletotrichum species)
3. **Leaf Spot** - Concentric rings and brownish spots 
4. **Leaf Blight** - Irregular spots with chlorosis and curling (Alternaria)
5. **Mosaic Virus** - Mottled patterns and chlorotic patches (YMV)
6. **Mild Mosaic** - Mild mottling symptoms (YMMV)
7. **Bacterial Spot** - Angular water-soaked lesions with halos
8. **Bacilliform Virus** - Mild symptoms, chlorosis or mosaic (DBV)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install ChromeDriver:
   - Download from: https://chromedriver.chromium.org/
   - Add to system PATH

## Usage

### 1. Scrape Images
```bash
python dataset_scraper.py
```

This will:
- Create organized folders for each disease category
- Search and download images from Google Images
- Save metadata about each download
- Generate a comprehensive dataset report

### 2. Process Dataset
```bash
python dataset_utils.py
```

This will:
- Validate all downloaded images
- Remove corrupted/invalid files
- Apply data augmentation to balance classes
- Create train/validation splits (80/20)
- Resize images to 224x224 for ML training
- Generate labels and class mapping files

## Dataset Structure

After processing, your dataset will be organized as:

```
yam_disease_dataset/
├── train/
│   ├── healthy/
│   ├── anthracnose/
│   ├── leaf_spot/
│   ├── leaf_blight/
│   ├── mosaic_virus/
│   ├── mild_mosaic/
│   ├── bacterial_spot/
│   └── bacilliform_virus/
├── val/
│   ├── healthy/
│   └── ... (same categories)
├── metadata/
│   └── category_metadata.json files
├── labels.txt
├── class_mapping.json
└── dataset_report.json
```

## Configuration

You can modify these parameters in the scripts:

- `max_images_per_query`: Images to download per search term (default: 30)
- `target_count`: Target images per category after augmentation (default: 150)
- `train_ratio`: Training/validation split ratio (default: 0.8)
- `target_size`: Image resize dimensions (default: 224x224)

## Data Augmentation

The processor applies random augmentations to balance classes:
- Rotation (-15° to +15°)
- Brightness adjustment (0.8-1.2x)
- Contrast adjustment (0.8-1.2x)
- Color saturation (0.8-1.2x)
- Horizontal/vertical flipping

## Usage in Flutter App

To integrate with your Flutter yam disease detection app:

1. Train your model using this dataset
2. Convert model to TensorFlow Lite format
3. Add the model to Flutter assets:

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/yam_disease_model.tflite
    - assets/models/labels.txt
```

## Ethical Considerations

- Images are collected from public search results
- No copyrighted content is intentionally downloaded
- Use collected data responsibly and in accordance with applicable laws
- Consider data licensing for commercial applications

## Troubleshooting

1. **ChromeDriver Issues**: Ensure ChromeDriver version matches your Chrome browser
2. **Low Image Counts**: Some disease categories may have limited online images
3. **Network Errors**: Script includes retry logic and rate limiting
4. **Invalid Images**: Automatic validation removes corrupted files

## Next Steps

1. Run the scraper to collect initial images
2. Manually review and clean the dataset
3. Add your own high-quality images if available
4. Train your machine learning model
5. Integrate trained model into your Flutter app

This dataset will provide a solid foundation for training yam disease detection models with good coverage of the major disease categories affecting yam crops.
