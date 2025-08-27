# 🧪 Yam Disease Model Evaluation Guide

Complete guide to evaluate your Teachable Machine model with professional metrics: **F1 score, accuracy, precision, recall, and specificity**.

## 📊 Available Evaluation Methods

### 1. **Python Comprehensive Evaluation** (Recommended)
- Full scientific evaluation with visualizations
- Confusion matrices, ROC curves, detailed reports
- Professional metrics for research/publication

### 2. **Flutter In-App Metrics** 
- Real-time evaluation during app usage
- Live performance monitoring
- User-friendly metrics display

## 🚀 Method 1: Python Comprehensive Evaluation

### **Setup:**
```bash
# Install evaluation dependencies
pip install -r evaluation_requirements.txt

# Run comprehensive evaluation
python model_evaluation.py
```

### **What You Get:**
- **📊 Visual Charts** - Confusion matrix, F1-scores, precision/recall plots
- **📝 Detailed Reports** - Markdown and JSON format
- **🎯 Per-Class Metrics** - Individual disease performance
- **📈 Overall Performance** - Complete model assessment

### **Generated Files:**
```
evaluation_results/
├── model_evaluation_charts.png    # Visual metrics charts
├── evaluation_report.md           # Human-readable report  
└── evaluation_report.json         # Machine-readable data
```

### **Metrics Included:**
```
📊 OVERALL METRICS:
✅ Accuracy - Overall correct predictions
✅ Precision (Macro/Weighted) - Positive prediction accuracy  
✅ Recall (Macro/Weighted) - Actual positive detection rate
✅ F1-Score (Macro/Weighted) - Harmonic mean of precision/recall
✅ Specificity - True negative rate per class

🎯 PER-CLASS METRICS:
✅ Individual F1, Precision, Recall, Specificity for each disease
✅ Support - Number of samples per class
✅ Confusion Matrix - Detailed prediction breakdown
```

## 🔬 Method 2: Flutter In-App Evaluation

### **Integration:**

1. **Import the metrics module** in your `main.dart`:
```dart
import 'model_metrics.dart';
```

2. **Add to your classification results** (in `_pickImage` method):
```dart
// After getting classification result
if (_classificationResult != null) {
  // For testing, you can manually add ground truth
  // In production, you'd get this from user feedback or validation set
  int predictedClass = _labels.indexOf(_classificationResult!['topPrediction']['label']);
  int actualClass = 0; // This would come from your validation data
  
  globalMetrics.addPrediction(predictedClass, actualClass);
}
```

3. **Display metrics** with a new button:
```dart
FloatingActionButton(
  onPressed: _showMetrics,
  child: Icon(Icons.analytics),
)

void _showMetrics() {
  String report = globalMetrics.generateMetricsReport();
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text('Model Performance'),
      content: SingleChildScrollView(
        child: Text(report, style: TextStyle(fontFamily: 'monospace')),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(context), child: Text('Close'))
      ],
    ),
  );
}
```

## 📈 Understanding Your Metrics

### **Key Metrics Explained:**

| Metric | Formula | What It Means | Good Value |
|--------|---------|---------------|------------|
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | Overall correctness | >0.85 |
| **Precision** | `TP/(TP+FP)` | Of predicted positives, how many were right | >0.80 |
| **Recall** | `TP/(TP+FN)` | Of actual positives, how many were found | >0.80 |
| **F1-Score** | `2×(Precision×Recall)/(Precision+Recall)` | Balanced precision/recall | >0.75 |
| **Specificity** | `TN/(TN+FP)` | Of actual negatives, how many were correctly identified | >0.90 |

### **Performance Grades:**
- 🏆 **0.90+** - Excellent (Research quality)
- ✅ **0.80-0.89** - Very Good (Production ready)
- 👍 **0.70-0.79** - Good (Acceptable)
- ⚠️  **0.60-0.69** - Fair (Needs improvement)
- ❌ **<0.60** - Poor (Retrain model)

## 🎯 Model Improvement Tips

### **If F1-Score < 0.75:**
1. **More Training Data** - Add more images per disease class
2. **Better Data Quality** - Remove low-quality/mislabeled images
3. **Balanced Dataset** - Ensure equal samples per disease
4. **Data Augmentation** - Use rotation, brightness, contrast variations

### **If Specific Diseases Perform Poorly:**
1. **Visual Similarity** - Some diseases look similar (leaf_spot vs leaf_blight)
2. **More Examples** - Focus on collecting more data for poor-performing classes
3. **Feature Engineering** - Enhance distinctive characteristics in training

### **For Production Use:**
- **Minimum F1-Score:** 0.75 for agricultural applications
- **Critical Diseases** (anthracnose, mosaic_virus): Aim for >0.85 recall
- **False Positives vs False Negatives:** Prioritize based on treatment cost

## 📋 Validation Dataset Setup

### **Recommended Structure:**
```
yam_disease_dataset/
├── train/ (80% - for Teachable Machine training)
│   ├── healthy/ (100+ images)
│   ├── anthracnose/ (100+ images)
│   └── ... (other diseases)
├── val/ (20% - for evaluation)
│   ├── healthy/ (25+ images)
│   ├── anthracnose/ (25+ images)
│   └── ... (other diseases)
```

### **Creating Validation Split:**
```python
# Use your dataset_utils.py
python dataset_utils.py  # This creates train/val splits automatically
```

## 🚀 Running Full Evaluation

### **Complete Workflow:**
```bash
# 1. Ensure you have validation data
ls yam_disease_dataset/val/

# 2. Install requirements  
pip install -r evaluation_requirements.txt

# 3. Run evaluation
python model_evaluation.py

# 4. Check results
open evaluation_results/model_evaluation_charts.png
open evaluation_results/evaluation_report.md
```

### **Expected Output:**
```
🔬 Starting Model Evaluation...
📊 Running predictions...
Progress: 200/400 (50.0%)
✅ Predictions complete: 400 samples

📈 Calculating metrics...
✅ Metrics calculated successfully

🎨 Creating visualizations...
📊 Visualizations saved to: evaluation_results/

📝 Generating evaluation report...
✅ Reports saved to: evaluation_results/

🎉 MODEL EVALUATION COMPLETE
📊 Overall Accuracy: 0.8750 (87.50%)
🎯 Macro F1-Score: 0.8234
```

## 📊 Interpreting Results

### **Confusion Matrix:**
- **Diagonal values** = Correct predictions
- **Off-diagonal** = Misclassifications
- **Dark blue squares** = High numbers

### **Per-Class F1-Scores:**
- **Green bars** = Good performance (>0.8)
- **Yellow bars** = Fair performance (0.6-0.8)
- **Red bars** = Poor performance (<0.6)

### **Precision vs Recall Plot:**
- **Top-right corner** = Best performing diseases
- **Bottom-left corner** = Worst performing diseases
- **Diagonal line** = Balanced precision/recall

## 🎖️ Professional Reporting

For research papers or professional use:

```markdown
## Model Performance

Our Teachable Machine model achieved an overall accuracy of 87.5% 
with a macro F1-score of 0.823 across 8 yam disease classes. 

The model demonstrated excellent performance for healthy leaf 
detection (F1=0.91) and anthracnose identification (F1=0.87), 
while showing moderate performance for viral diseases 
(mosaic_virus F1=0.74, mild_mosaic F1=0.69).

Specificity was consistently high (>0.92) across all classes, 
indicating low false positive rates suitable for agricultural 
decision-making applications.
```

Your yam disease detection model evaluation system is now complete! 🌱📊🎯
