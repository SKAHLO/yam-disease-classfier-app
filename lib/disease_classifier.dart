import 'dart:typed_data';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

class YamDiseaseClassifier {
  static const String modelPath = 'assets/models/yam_disease_model.tflite';
  static const String labelsPath = 'assets/models/labels.txt';
  
  late Interpreter _interpreter;
  late List<String> _labels;
  bool _isModelLoaded = false;
  
  // Disease information map
  static const Map<String, Map<String, String>> diseaseInfo = {
    'healthy': {
      'description': 'Healthy yam leaf with no disease symptoms',
      'treatment': 'Continue good agricultural practices',
      'severity': 'None'
    },
    'anthracnose': {
      'description': 'Dark spots with yellow halos caused by Colletotrichum',
      'treatment': 'Apply fungicide, improve drainage, remove infected leaves',
      'severity': 'High'
    },
    'leaf_spot': {
      'description': 'Concentric rings and brownish spots on leaves',
      'treatment': 'Use copper-based fungicide, ensure proper spacing',
      'severity': 'Medium'
    },
    'leaf_blight': {
      'description': 'Irregular spots with chlorosis and leaf curling',
      'treatment': 'Apply Mancozeb, improve air circulation',
      'severity': 'High'
    },
    'mosaic_virus': {
      'description': 'Mottled patterns and chlorotic patches (YMV)',
      'treatment': 'Remove infected plants, control aphid vectors',
      'severity': 'Very High'
    },
    'mild_mosaic': {
      'description': 'Mild mottling symptoms (YMMV)',
      'treatment': 'Monitor closely, remove if symptoms worsen',
      'severity': 'Medium'
    },
    'bacterial_spot': {
      'description': 'Angular water-soaked lesions with yellow halos',
      'treatment': 'Apply copper bactericide, avoid overhead watering',
      'severity': 'High'
    },
    'bacilliform_virus': {
      'description': 'Mild chlorosis or mosaic symptoms (DBV)',
      'treatment': 'Remove infected plants, use clean planting material',
      'severity': 'Medium'
    }
  };
  
  Future<void> loadModel() async {
    try {
      // Load the model
      _interpreter = await Interpreter.fromAsset(modelPath);
      
      // Load labels
      final labelsData = await rootBundle.loadString(labelsPath);
      _labels = labelsData.split('\n').where((label) => label.isNotEmpty).toList();
      
      _isModelLoaded = true;
      print('Model loaded successfully with ${_labels.length} classes');
    } catch (e) {
      print('Error loading model: $e');
      throw Exception('Failed to load disease classification model');
    }
  }
  
  Future<Map<String, dynamic>> classifyImage(File imageFile) async {
    if (!_isModelLoaded) {
      await loadModel();
    }
    
    try {
      // Read and preprocess image
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Could not decode image');
      }
      
      // Resize image to model input size (224x224)
      final resized = img.copyResize(image, width: 224, height: 224);
      
      // Convert to float32 and normalize
      final input = _imageToByteListFloat32(resized);
      
      // Prepare output buffer [1, number_of_classes]
      final output = List.generate(1, (_) => List.filled(_labels.length, 0.0));
      
      // Run inference
      _interpreter.run([input], output);
      
      // Process results
      final predictions = output[0];
      final results = <Map<String, dynamic>>[];
      
      for (int i = 0; i < predictions.length && i < _labels.length; i++) {
        final confidence = predictions[i];
        final label = _labels[i];
        
        results.add({
          'label': label,
          'confidence': confidence,
          'percentage': (confidence * 100).toStringAsFixed(1),
        });
      }
      
      // Sort by confidence
      results.sort((a, b) => b['confidence'].compareTo(a['confidence']));
      
      final topResult = results.first;
      final diseaseKey = topResult['label'].toString().toLowerCase();
      
      return {
        'predictions': results,
        'topPrediction': topResult,
        'diseaseInfo': diseaseInfo[diseaseKey] ?? diseaseInfo['healthy'],
        'isHealthy': diseaseKey == 'healthy',
        'needsTreatment': topResult['confidence'] > 0.7 && diseaseKey != 'healthy',
      };
      
    } catch (e) {
      print('Classification error: $e');
      throw Exception('Failed to classify image: $e');
    }
  }
  
  Float32List _imageToByteListFloat32(img.Image image) {
    final convertedBytes = Float32List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    
    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        final pixel = image.getPixel(j, i);
        
        // Extract RGB values from pixel (newer image package format)
        final r = pixel.r;
        final g = pixel.g;
        final b = pixel.b;
        
        // Normalize RGB values to [0,1]
        convertedBytes[pixelIndex++] = r / 255.0;
        convertedBytes[pixelIndex++] = g / 255.0;
        convertedBytes[pixelIndex++] = b / 255.0;
      }
    }
    
    return convertedBytes;
  }
  
  void dispose() {
    _interpreter.close();
  }
}
