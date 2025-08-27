import 'dart:math';

class ModelMetrics {
  // Storage for predictions and true labels
  List<int> predictions = [];
  List<int> trueLabels = [];
  List<String> classNames = [
    'healthy',
    'anthracnose',
    'leaf_spot', 
    'leaf_blight',
    'mosaic_virus',
    'mild_mosaic',
    'bacterial_spot',
    'bacilliform_virus'
  ];
  
  void addPrediction(int predicted, int actual) {
    predictions.add(predicted);
    trueLabels.add(actual);
  }
  
  void clearMetrics() {
    predictions.clear();
    trueLabels.clear();
  }
  
  Map<String, dynamic> calculateMetrics() {
    if (predictions.isEmpty || trueLabels.isEmpty) {
      return {'error': 'No predictions available'};
    }
    
    int numClasses = classNames.length;
    
    // Create confusion matrix
    List<List<int>> confusionMatrix = List.generate(
      numClasses, (_) => List.filled(numClasses, 0)
    );
    
    for (int i = 0; i < predictions.length; i++) {
      confusionMatrix[trueLabels[i]][predictions[i]]++;
    }
    
    // Calculate per-class metrics
    Map<String, Map<String, double>> perClassMetrics = {};
    List<double> precisions = [];
    List<double> recalls = [];
    List<double> f1Scores = [];
    List<double> specificities = [];
    
    for (int classIdx = 0; classIdx < numClasses; classIdx++) {
      String className = classNames[classIdx];
      
      // True Positives, False Positives, False Negatives, True Negatives
      int tp = confusionMatrix[classIdx][classIdx];
      int fp = 0;
      int fn = 0;
      int tn = 0;
      
      for (int i = 0; i < numClasses; i++) {
        for (int j = 0; j < numClasses; j++) {
          if (i == classIdx && j != classIdx) {
            fn += confusionMatrix[i][j]; // False Negatives
          } else if (i != classIdx && j == classIdx) {
            fp += confusionMatrix[i][j]; // False Positives
          } else if (i != classIdx && j != classIdx) {
            tn += confusionMatrix[i][j]; // True Negatives
          }
        }
      }
      
      // Calculate metrics
      double precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0;
      double recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
      double f1Score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
      double specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0.0;
      
      precisions.add(precision);
      recalls.add(recall);
      f1Scores.add(f1Score);
      specificities.add(specificity);
      
      perClassMetrics[className] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1Score,
        'specificity': specificity,
        'support': (tp + fn).toDouble(),
      };
    }
    
    // Calculate overall metrics
    double accuracy = _calculateAccuracy();
    double macroPrecision = precisions.reduce((a, b) => a + b) / precisions.length;
    double macroRecall = recalls.reduce((a, b) => a + b) / recalls.length;
    double macroF1 = f1Scores.reduce((a, b) => a + b) / f1Scores.length;
    double macroSpecificity = specificities.reduce((a, b) => a + b) / specificities.length;
    
    // Weighted metrics (by support)
    double totalSupport = predictions.length.toDouble();
    double weightedPrecision = 0;
    double weightedRecall = 0;
    double weightedF1 = 0;
    
    for (String className in classNames) {
      double support = perClassMetrics[className]!['support']!;
      double weight = support / totalSupport;
      
      weightedPrecision += perClassMetrics[className]!['precision']! * weight;
      weightedRecall += perClassMetrics[className]!['recall']! * weight;
      weightedF1 += perClassMetrics[className]!['f1_score']! * weight;
    }
    
    return {
      'overall_metrics': {
        'accuracy': accuracy,
        'macro_precision': macroPrecision,
        'macro_recall': macroRecall,
        'macro_f1_score': macroF1,
        'macro_specificity': macroSpecificity,
        'weighted_precision': weightedPrecision,
        'weighted_recall': weightedRecall,
        'weighted_f1_score': weightedF1,
        'total_samples': predictions.length,
      },
      'per_class_metrics': perClassMetrics,
      'confusion_matrix': confusionMatrix,
      'class_names': classNames,
    };
  }
  
  double _calculateAccuracy() {
    if (predictions.isEmpty || trueLabels.isEmpty) return 0.0;
    
    int correct = 0;
    for (int i = 0; i < predictions.length; i++) {
      if (predictions[i] == trueLabels[i]) {
        correct++;
      }
    }
    
    return correct / predictions.length;
  }
  
  String generateMetricsReport() {
    Map<String, dynamic> metrics = calculateMetrics();
    
    if (metrics.containsKey('error')) {
      return 'No metrics available - make some predictions first!';
    }
    
    StringBuffer report = StringBuffer();
    
    report.writeln('🎯 Model Performance Report');
    report.writeln('=' * 40);
    
    Map<String, dynamic> overall = metrics['overall_metrics'];
    report.writeln('📊 Overall Metrics:');
    report.writeln('• Accuracy: ${(overall['accuracy'] * 100).toStringAsFixed(2)}%');
    report.writeln('• Macro F1-Score: ${overall['macro_f1_score'].toStringAsFixed(3)}');
    report.writeln('• Macro Precision: ${overall['macro_precision'].toStringAsFixed(3)}');
    report.writeln('• Macro Recall: ${overall['macro_recall'].toStringAsFixed(3)}');
    report.writeln('• Macro Specificity: ${overall['macro_specificity'].toStringAsFixed(3)}');
    report.writeln('• Total Samples: ${overall['total_samples']}');
    
    report.writeln('\n🔍 Per-Class Performance:');
    Map<String, Map<String, double>> perClass = metrics['per_class_metrics'];
    
    for (String className in classNames) {
      if (perClass.containsKey(className)) {
        Map<String, double> classMetrics = perClass[className]!;
        String displayName = className.replaceAll('_', ' ').toUpperCase();
        
        report.writeln('$displayName:');
        report.writeln('  Precision: ${classMetrics['precision']!.toStringAsFixed(3)}');
        report.writeln('  Recall: ${classMetrics['recall']!.toStringAsFixed(3)}');
        report.writeln('  F1-Score: ${classMetrics['f1_score']!.toStringAsFixed(3)}');
        report.writeln('  Specificity: ${classMetrics['specificity']!.toStringAsFixed(3)}');
        report.writeln('  Support: ${classMetrics['support']!.toInt()}');
        report.writeln('');
      }
    }
    
    // Performance assessment
    double overallF1 = overall['macro_f1_score'];
    report.writeln('🎖️  Performance Assessment:');
    if (overallF1 >= 0.9) {
      report.writeln('EXCELLENT - Outstanding model performance!');
    } else if (overallF1 >= 0.8) {
      report.writeln('VERY GOOD - High-quality predictions');
    } else if (overallF1 >= 0.7) {
      report.writeln('GOOD - Solid performance');
    } else if (overallF1 >= 0.6) {
      report.writeln('FAIR - Room for improvement');
    } else {
      report.writeln('NEEDS IMPROVEMENT - Consider model retraining');
    }
    
    return report.toString();
  }
  
  // Method to get the best and worst performing classes
  Map<String, String> getPerformanceHighlights() {
    Map<String, dynamic> metrics = calculateMetrics();
    
    if (metrics.containsKey('error')) {
      return {'best': 'N/A', 'worst': 'N/A'};
    }
    
    Map<String, Map<String, double>> perClass = metrics['per_class_metrics'];
    
    String bestClass = '';
    String worstClass = '';
    double bestF1 = -1;
    double worstF1 = 2;
    
    for (String className in perClass.keys) {
      double f1 = perClass[className]!['f1_score']!;
      
      if (f1 > bestF1) {
        bestF1 = f1;
        bestClass = className;
      }
      
      if (f1 < worstF1) {
        worstF1 = f1;
        worstClass = className;
      }
    }
    
    return {
      'best': '${bestClass.replaceAll('_', ' ').toUpperCase()} (${(bestF1 * 100).toStringAsFixed(1)}%)',
      'worst': '${worstClass.replaceAll('_', ' ').toUpperCase()} (${(worstF1 * 100).toStringAsFixed(1)}%)',
    };
  }
}

// Singleton instance for global access
final ModelMetrics globalMetrics = ModelMetrics();
