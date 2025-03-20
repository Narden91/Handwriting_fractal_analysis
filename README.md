# ✍️ Handwriting Fractal Analyzer

A python tool for analyzing handwriting using advanced fractal geometry and morphological analysis techniques.

## 🌟 Features

### 📊 Basic Fractal Features

**Fractal Dimension**
- 🔍 **Calculation**: Negative slope of log-log relationship between box size and box count
- 💡 **Meaning**: Quantifies complexity and space-filling capacity of handwriting patterns
- 🎯 **Significance**: Higher values indicate more complex, irregular writing with fine details

**Box Counts**
- 🔍 **Calculation**: Number of boxes containing handwriting at each scale
- 💡 **Meaning**: Measures how handwriting occupies space at different resolutions
- 🎯 **Significance**: Captures multi-scale spatial distribution characteristics

**Density Measures**
- 🔍 **Calculation**: Ratio of boxes containing handwriting to total possible boxes
- 💡 **Meaning**: Quantifies the concentration of handwriting elements
- 🎯 **Significance**: Differentiates between dense, compact writing and sparse patterns

### 🧩 Lacunarity Features

**Standard Lacunarity**
- 🔍 **Calculation**: Ratio of variance to squared mean of box densities
- 💡 **Meaning**: Measures the "gappiness" or heterogeneity of handwriting
- 🎯 **Significance**: Captures variations in spacing and distribution of strokes

**Gap Ratio**
- 🔍 **Calculation**: Proportion of empty boxes at each scale
- 💡 **Meaning**: Quantifies prevalence of empty spaces in writing
- 🎯 **Significance**: Indicates writing style related to spacing and openness

### 🌈 Multifractal Spectrum Features

**q-Order Moments**
- 🔍 **Calculation**: Generalized dimensions for different q values using partition functions
- 💡 **Meaning**: Characterizes different scaling regimes within handwriting
- 🎯 **Significance**: Captures both dominant and subtle patterns simultaneously

**Spectrum Width**
- 🔍 **Calculation**: Difference between maximum and minimum generalized dimensions
- 💡 **Meaning**: Indicates strength of multifractality in handwriting
- 🎯 **Significance**: Distinguishes between monofractal and multifractal writing styles

### 🧭 Directional Fractal Features

**Directional Dimensions**
- 🔍 **Calculation**: Fractal dimensions along specific angles (0°, 45°, 90°, 135°)
- 💡 **Meaning**: Quantifies complexity in different orientations
- 🎯 **Significance**: Reveals preferred writing directions and stroke patterns

**Anisotropy Index**
- 🔍 **Calculation**: Standard deviation of directional dimensions normalized by mean
- 💡 **Meaning**: Measures directional dependency of handwriting patterns
- 🎯 **Significance**: Differentiates between uniform and direction-dependent writing

### ✒️ Stroke Analysis Features

**Stroke Width Statistics**
- 🔍 **Calculation**: Mean, variance, min/max of widths measured along stroke skeleton
- 💡 **Meaning**: Characterizes thickness and variability of strokes
- 🎯 **Significance**: Correlates with writing instrument, pressure, and motor control

**Junction Analysis**
- 🔍 **Calculation**: Count and density of points where strokes intersect
- 💡 **Meaning**: Quantifies connectivity patterns in handwriting
- 🎯 **Significance**: Reveals writing complexity and characteristic connection patterns

### 🔄 Topological Features

**Persistence Measures**
- 🔍 **Calculation**: Statistics on birth/death of connected components across scales
- 💡 **Meaning**: Captures how structural features persist as resolution changes
- 🎯 **Significance**: Provides scale-invariant descriptors resistant to minor variations

**Euler Characteristics**
- 🔍 **Calculation**: Connected components minus holes at each scale
- 💡 **Meaning**: Fundamental topological invariant describing structure
- 🎯 **Significance**: Quantifies connectivity patterns independent of deformation

### 📐 Spatial Distribution Features

**Centroid and Balance**
- 🔍 **Calculation**: Center of mass and distribution around it
- 💡 **Meaning**: Quantifies overall positioning and balance of writing
- 🎯 **Significance**: Relates to page organization and spatial awareness

**Spatial Entropy**
- 🔍 **Calculation**: Shannon entropy of grid cell densities
- 💡 **Meaning**: Quantifies randomness in spatial distribution
- 🎯 **Significance**: Distinguishes between organized and chaotic spatial arrangements

## 🛠️ Tools Used in this Project

* [hydra](https://hydra.cc/): Advanced configuration management
* [OpenCV](https://opencv.org/): Image processing and analysis
* [NumPy](https://numpy.org/): Numerical computations
* [scikit-learn](https://scikit-learn.org/): Machine learning tools
* [rich](https://github.com/Textualize/rich): Terminal formatting and display

## 📂 Project Structure

```bash
.
├── config                      
│   └── main.yaml                   # Main configuration file
├── data                            # Directory to store input data
│   └── TASK*                       # Task-specific data folders
├── docs                            # Documentation for your project
├── .gitignore                      # Ignore files that cannot commit to Git
├── main.py                         # Main script entry
├── models                          # Store trained models
├── pyproject.toml                  # Configure code style and tools
├── README.md                       # Project description
├── requirements.txt                # Package dependencies
├── results                         # Analysis output directory
└── src                             # Source code
    ├── __init__.py                 # Make src a Python module
    ├── fractal_analyzer
    │   ├── __init__.py             # Make fractal_analyzer a Python module
    │   └── fractal_analyzer.py     # Core fractal analysis implementation
    └── utils.py                    # Helper functions
```

## 🚀 Getting Started

### 🔧 Set up the Environment

1. Create the virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:

- For Linux/MacOS:
```bash
source venv/bin/activate
```
- For Windows:
```bash
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 🏃‍♂️ Running the Analysis

Run the main script:
```bash
python main.py
```

#### 🎛️ Configuration Options

You can override configuration values:
```bash
python main.py data.path=path/to/your/images
```

Adjust verbosity level:
```bash
python main.py display.verbosity=2
```

## 📈 Analysis Process

1. 📥 **Input**: The system takes handwriting image samples as input.
2. 🔄 **Preprocessing**: Images are binarized and prepared for analysis.
3. 📊 **Fractal Analysis**: Multiple fractal and morphological measurements are applied.
4. 🧮 **Feature Extraction**: Comprehensive feature vectors are created for each image.
5. 💾 **Output**: Results are saved to CSV files for further analysis.

## 📝 Example Output

Each analyzed image produces a rich feature set including:
- Fractal dimensions at multiple scales
- Lacunarity measures of spatial heterogeneity
- Multifractal spectrum properties
- Directional analysis results
- Stroke characteristics
- Topological features
- Spatial distribution metrics

## 🔬 Advanced Usage

### Extending the Analysis

To add custom feature extractors:

1. Create a new method in the `FractalAnalyzer` class
2. Update the `analyze_image` method to include your new features
3. Add appropriate configuration in the `main.yaml` file

### Custom Visualization

The system supports custom visualization through the rich library:
```bash
python main.py display.theme=dracula display.verbosity=2
```