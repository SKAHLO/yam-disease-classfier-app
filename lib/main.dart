import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'disease_classifier.dart';

void main() {
  runApp(const YamDiseaseApp());
}

class YamDiseaseApp extends StatelessWidget {
  const YamDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Yam Disease Detector',
      debugShowCheckedModeBanner: false, // Remove debug banner
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4CAF50),
          brightness: Brightness.light,
        ),
        appBarTheme: const AppBarTheme(
          elevation: 0,
          centerTitle: true,
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.white,
        ),
      ),
      home: const YamDiseaseDetector(),
    );
  }
}

class YamDiseaseDetector extends StatefulWidget {
  const YamDiseaseDetector({super.key});

  @override
  State<YamDiseaseDetector> createState() => _YamDiseaseDetectorState();
}

class _YamDiseaseDetectorState extends State<YamDiseaseDetector> {
  final ImagePicker _picker = ImagePicker();
  final YamDiseaseClassifier _classifier = YamDiseaseClassifier();
  
  File? _image;
  Map<String, dynamic>? _classificationResult;
  bool _isLoading = false;
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
    
    // Set your Teachable Machine model URL for direct API integration
    _classifier.setTeachableMachineUrl('https://teachablemachine.withgoogle.com/models/99_E2CS7s/model.json');
  }

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
    });
    
    try {
      await _classifier.loadModel();
      if (mounted) {
        setState(() {
          _isModelLoaded = true;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading model: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? pickedFile = await _picker.pickImage(
      source: source,
      maxWidth: 800,
      maxHeight: 800,
      imageQuality: 85,
    );

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _classificationResult = null;
        _isLoading = true;
      });

      try {
        final result = await _classifier.classifyImage(_image!);
        if (mounted) {
          setState(() {
            _classificationResult = result;
          });
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Classification error: $e')),
          );
        }
      } finally {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF4CAF50),
              Color(0xFF8BC34A),
              Color(0xFF66BB6A),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Custom App Bar
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.eco,
                        color: Colors.white,
                        size: 24,
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Yam Disease',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            'AI-Powered Detector',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (_isModelLoaded)
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.9),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.smart_toy,
                              color: Colors.green[600],
                              size: 16,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              'AI Ready',
                              style: TextStyle(
                                color: Colors.green[700],
                                fontWeight: FontWeight.w600,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
              
              // Main Content
              Expanded(
                child: Container(
                  margin: const EdgeInsets.only(top: 20),
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.vertical(
                      top: Radius.circular(30),
                    ),
                  ),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        // Status Card
                        _buildStatusCard(),
                        
                        const SizedBox(height: 24),
                        
                        // Camera Buttons
                        _buildCameraButtons(),
                        
                        const SizedBox(height: 24),
                        
                        // Image Display
                        if (_image != null) _buildImageDisplay(),
                        
                        // Loading Indicator
                        if (_isLoading) _buildLoadingIndicator(),
                        
                        // Classification Results
                        if (_classificationResult != null && !_isLoading)
                          _buildResultsCard(),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildStatusCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            _isModelLoaded ? Colors.green[50]! : Colors.orange[50]!,
            _isModelLoaded ? Colors.green[100]! : Colors.orange[100]!,
          ],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _isModelLoaded ? Colors.green[200]! : Colors.orange[200]!,
          width: 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: _isModelLoaded ? Colors.green : Colors.orange,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  _isModelLoaded ? Icons.psychology : Icons.hourglass_empty,
                  color: Colors.white,
                  size: 24,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _isModelLoaded ? 'AI System Ready' : 'Initializing...',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: _isModelLoaded ? Colors.green[800] : Colors.orange[800],
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _isModelLoaded 
                          ? 'Teachable Machine Enhanced'
                          : 'Loading classification system',
                      style: TextStyle(
                        color: Colors.grey[600],
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (_classificationResult != null && _classificationResult!['usingModel'] != null)
            Container(
              margin: const EdgeInsets.only(top: 12),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.green[600],
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.verified,
                    color: Colors.white,
                    size: 16,
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'Teachable Machine Active',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
  
  Widget _buildCameraButtons() {
    return Row(
      children: [
        Expanded(
          child: _buildActionButton(
            icon: Icons.camera_alt,
            label: 'Take Photo',
            gradient: const LinearGradient(
              colors: [Color(0xFF42A5F5), Color(0xFF1E88E5)],
            ),
            onPressed: _isModelLoaded && !_isLoading
                ? () => _pickImage(ImageSource.camera)
                : null,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: _buildActionButton(
            icon: Icons.photo_library,
            label: 'Gallery',
            gradient: const LinearGradient(
              colors: [Color(0xFF66BB6A), Color(0xFF4CAF50)],
            ),
            onPressed: _isModelLoaded && !_isLoading
                ? () => _pickImage(ImageSource.gallery)
                : null,
          ),
        ),
      ],
    );
  }
  
  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required LinearGradient gradient,
    required VoidCallback? onPressed,
  }) {
    return Container(
      decoration: BoxDecoration(
        gradient: onPressed != null ? gradient : LinearGradient(
          colors: [Colors.grey[300]!, Colors.grey[400]!],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: onPressed != null ? [
          BoxShadow(
            color: gradient.colors.first.withValues(alpha: 0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ] : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onPressed,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  icon,
                  color: Colors.white,
                  size: 24,
                ),
                const SizedBox(width: 12),
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
  
  Widget _buildImageDisplay() {
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Stack(
          children: [
            Image.file(
              _image!,
              width: double.infinity,
              height: 300,
              fit: BoxFit.cover,
            ),
            if (_isLoading)
              Container(
                width: double.infinity,
                height: 300,
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.3),
                ),
                child: const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                      SizedBox(height: 16),
                      Text(
                        'Analyzing with AI...',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildLoadingIndicator() {
    if (_image != null) return const SizedBox.shrink();
    
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 40),
      child: Column(
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF4CAF50), Color(0xFF8BC34A)],
              ),
              borderRadius: BorderRadius.circular(40),
            ),
            child: const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                strokeWidth: 3,
              ),
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'AI is analyzing your image...',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w500,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'This may take a few seconds',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsCard() {
    final result = _classificationResult!;
    final topPrediction = result['topPrediction'];
    final diseaseInfo = result['diseaseInfo'];
    final isHealthy = result['isHealthy'] ?? false;
    final needsTreatment = result['needsTreatment'] ?? false;
    
    return Container(
      margin: const EdgeInsets.only(top: 24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: isHealthy 
              ? [Colors.green[50]!, Colors.green[100]!]
              : [Colors.orange[50]!, Colors.red[50]!],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: (isHealthy ? Colors.green : Colors.orange).withValues(alpha: 0.2),
            blurRadius: 15,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header with animated result
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: isHealthy 
                            ? [Colors.green[400]!, Colors.green[600]!]
                            : [Colors.orange[400]!, Colors.red[500]!],
                      ),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      isHealthy ? Icons.eco : Icons.warning_rounded,
                      color: Colors.white,
                      size: 28,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _formatDiseaseLabel(topPrediction['label']),
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: isHealthy ? Colors.green[800] : Colors.red[800],
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: isHealthy ? Colors.green[100] : Colors.orange[100],
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            '${topPrediction['percentage']}% Confidence',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: isHealthy ? Colors.green[700] : Colors.orange[700],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Disease Information Cards
            _buildInfoCard('Description', diseaseInfo['description'], Icons.info_outline),
            const SizedBox(height: 12),
            _buildInfoCard('Severity Level', diseaseInfo['severity'], Icons.priority_high),
            const SizedBox(height: 12),
            _buildInfoCard('Treatment', diseaseInfo['treatment'], Icons.local_hospital),
            
            // Treatment Alert
            if (needsTreatment) ...[
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.red[100]!, Colors.red[50]!],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.red[200]!, width: 2),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.red,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(
                        Icons.medical_services,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'Immediate Treatment Required!\nThis plant shows clear signs of disease.',
                        style: TextStyle(
                          color: Colors.red,
                          fontWeight: FontWeight.w600,
                          fontSize: 14,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            // All Predictions
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.analytics,
                        color: Colors.blue[600],
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'Detailed Analysis',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ...((result['predictions'] as List).take(4).map((pred) => 
                    _buildPredictionRow(pred['label'], pred['percentage'])
                  )),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildInfoCard(String title, String? content, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            icon,
            color: Colors.grey[600],
            size: 20,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: Colors.grey[700],
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  content ?? 'N/A',
                  style: const TextStyle(
                    fontSize: 14,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildPredictionRow(String label, String percentage) {
    final confidence = double.tryParse(percentage.replaceAll('%', '')) ?? 0;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: Text(
              _formatDiseaseLabel(label),
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            flex: 2,
            child: Container(
              height: 8,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(4),
              ),
              child: FractionallySizedBox(
                alignment: Alignment.centerLeft,
                widthFactor: confidence / 100,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: confidence > 50 
                          ? [Colors.green[400]!, Colors.green[600]!]
                          : [Colors.blue[400]!, Colors.blue[600]!],
                    ),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            percentage,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Colors.grey[600],
            ),
          ),
        ],
      ),
    );
  }
  
  String _formatDiseaseLabel(String label) {
    return label
        .split('_')
        .map((word) => word[0].toUpperCase() + word.substring(1))
        .join(' ');
  }



  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }
}
