# Model Files

Place your trained TensorFlow Lite model here:

## Required Files:
- `yam_disease_model.tflite` - Your trained model from Google Teachable Machine
- `labels.txt` - Class labels (already provided)

## How to Get Your Model:
1. Train your model at https://teachablemachine.withgoogle.com/
2. Use the images from your `yam_disease_dataset` folder
3. Export as "TensorFlow Lite"
4. Download and place `model.tflite` here as `yam_disease_model.tflite`

## Labels Order:
Make sure your Teachable Machine categories are in this order:
1. healthy
2. anthracnose
3. leaf_spot
4. leaf_blight
5. mosaic_virus
6. mild_mosaic
7. bacterial_spot
8. bacilliform_virus

## Fallback Mode:
If no model file is found, the app will use intelligent color/texture analysis as a fallback classification method.
