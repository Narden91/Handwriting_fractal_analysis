import numpy as np
import cv2
from pathlib import Path


class FractalAnalyzer:
    """
    A class for analyzing binary images using fractal and lacunarity measures
    with multi-scale box counting methods.
    """
    
    def __init__(self, box_sizes=None):
        """
        Initialize the FractalAnalyzer with specified box sizes.
        
        Parameters:
        -----------
        box_sizes : array-like, optional
            Box sizes to use for analysis. Default is [2, 4, 8, 16, 32].
        """
        self.box_sizes = np.array(box_sizes) if box_sizes is not None else np.array([2, 4, 8, 16, 32])
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for analysis.
        
        Parameters:
        -----------
        image_path : str or Path
            Path to the image file.
            
        Returns:
        --------
        binary_image : ndarray
            Preprocessed binary image.
        """
        # Convert to Path object if string
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Binarize using Otsu's method
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Perform morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        
        return binary_image
    
    def _box_counting(self, binary_image, box_size):
        """
        Apply box counting method for a specific box size.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
        box_size : int
            Size of the box for counting.
            
        Returns:
        --------
        box_count : int
            Number of boxes containing foreground pixels.
        box_densities : ndarray
            Array containing densities of foreground pixels in each box.
        """
        # Handle edge case: box size larger than image
        h, w = binary_image.shape
        if box_size > h or box_size > w:
            # Return an appropriate value when box size is too large
            total_pixels = np.sum(binary_image > 0)
            if total_pixels > 0:
                return 1, np.array([total_pixels])
            else:
                return 0, np.array([0])
                
        # Ensure image dimensions are divisible by box_size
        h_pad = (box_size - h % box_size) % box_size
        w_pad = (box_size - w % box_size) % box_size
        if h_pad > 0 or w_pad > 0:
            binary_image = np.pad(binary_image, ((0, h_pad), (0, w_pad)), mode='constant')
        
        h, w = binary_image.shape
        
        # Reshape to group pixels into boxes
        try:
            reshaped = binary_image.reshape(h // box_size, box_size, w // box_size, box_size)
        except ValueError as e:
            # Handle reshaping error
            raise ValueError(f"Error reshaping image for box size {box_size}: {e}")
        
        # Count foreground pixels in each box
        box_densities = np.sum(reshaped > 0, axis=(1, 3))
        
        # Count boxes with at least one foreground pixel
        box_count = np.sum(box_densities > 0)
        
        return box_count, box_densities
    
    def calculate_fractal_measures(self, binary_image):
        """
        Calculate fractal measures for all specified box sizes.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
            
        Returns:
        --------
        fractal_dict : dict
            Dictionary containing fractal measures with descriptive keys.
        """
        fractal_dict = {}
        h, w = binary_image.shape
        image_area = h * w
        
        for box_size in self.box_sizes:
            try:
                box_count, _ = self._box_counting(binary_image, box_size)
                
                # Raw box count - number of boxes containing handwriting
                fractal_dict[f"fractal_count_r{box_size}"] = int(box_count)
                
                # Box-size normalized count (scaling factor)
                if box_size <= min(h, w):  # Only calculate if box size is valid
                    max_possible_boxes = (h // box_size) * (w // box_size)
                    fractal_dict[f"fractal_density_r{box_size}"] = box_count / max_possible_boxes if max_possible_boxes > 0 else 0
                else:
                    fractal_dict[f"fractal_density_r{box_size}"] = 1.0 if box_count > 0 else 0.0
                
                # Log measures for fractal dimension calculation
                fractal_dict[f"fractal_log_count_r{box_size}"] = np.log(box_count) if box_count > 0 else 0
                fractal_dict[f"fractal_log_size_r{box_size}"] = np.log(box_size)
                
            except Exception as e:
                # Log error and set default values
                print(f"Error calculating fractal measures for box size {box_size}: {e}")
                fractal_dict[f"fractal_count_r{box_size}"] = 0
                fractal_dict[f"fractal_density_r{box_size}"] = 0
                fractal_dict[f"fractal_log_count_r{box_size}"] = 0
                fractal_dict[f"fractal_log_size_r{box_size}"] = np.log(box_size)
        
        # Calculate fractal dimension using log-log slope
        if len(self.box_sizes) >= 2:
            try:
                # Collect valid log measures
                log_sizes = []
                log_counts = []
                
                for box_size in self.box_sizes:
                    if (f"fractal_log_size_r{box_size}" in fractal_dict and 
                        f"fractal_log_count_r{box_size}" in fractal_dict):
                        
                        log_size = fractal_dict[f"fractal_log_size_r{box_size}"]
                        log_count = fractal_dict[f"fractal_log_count_r{box_size}"]
                        
                        if log_count > 0:  # Only use valid counts
                            log_sizes.append(log_size)
                            log_counts.append(log_count)
                
                if len(log_sizes) >= 2:
                    # Convert to numpy arrays
                    log_sizes = np.array(log_sizes)
                    log_counts = np.array(log_counts)
                    
                    # Calculate slope using linear regression
                    A = np.vstack([log_sizes, np.ones(len(log_sizes))]).T
                    slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
                    
                    # Fractal dimension is the negative of the slope
                    fractal_dict["fractal_dimension"] = -slope
                else:
                    fractal_dict["fractal_dimension"] = 0
                    
            except Exception as e:
                print(f"Error calculating fractal dimension: {e}")
                fractal_dict["fractal_dimension"] = 0
        else:
            fractal_dict["fractal_dimension"] = 0
        
        return fractal_dict
    
    def calculate_lacunarity_measures(self, binary_image):
        """
        Calculate lacunarity measures for all specified box sizes.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
            
        Returns:
        --------
        lacunarity_dict : dict
            Dictionary containing lacunarity measures with descriptive keys.
        """
        lacunarity_dict = {}
        
        for box_size in self.box_sizes:
            try:
                _, box_densities = self._box_counting(binary_image, box_size)
                
                # Flatten and filter to consider only non-empty boxes
                flat_densities = box_densities.flatten()
                
                # Standard lacunarity calculation
                if len(flat_densities) > 0:
                    # Calculate standard lacunarity: Λ(r) = σ²(r) / μ(r)²
                    # For all boxes (including empty ones)
                    mean_all = np.mean(flat_densities)
                    variance_all = np.var(flat_densities)
                    lacunarity_all = variance_all / (mean_all**2) if mean_all > 0 else 0
                    lacunarity_dict[f"lacunarity_all_r{box_size}"] = lacunarity_all
                    
                    # Gap ratio - proportion of empty boxes
                    total_boxes = len(flat_densities)
                    filled_boxes = np.sum(flat_densities > 0)
                    empty_boxes = total_boxes - filled_boxes
                    gap_ratio = empty_boxes / total_boxes if total_boxes > 0 else 0
                    lacunarity_dict[f"lacunarity_gap_ratio_r{box_size}"] = gap_ratio
                    
                    # Consider only filled boxes for these measures
                    filled_box_densities = flat_densities[flat_densities > 0]
                    if len(filled_box_densities) > 0:
                        mean = np.mean(filled_box_densities)
                        variance = np.var(filled_box_densities)
                        
                        # Standard lacunarity for filled boxes
                        lacunarity = variance / (mean**2) if mean > 0 else 0
                        lacunarity_dict[f"lacunarity_r{box_size}"] = lacunarity
                        
                        # Coefficient of variation - standardized measure of dispersion
                        cv = np.std(filled_box_densities) / mean if mean > 0 else 0
                        lacunarity_dict[f"lacunarity_cv_r{box_size}"] = cv
                        
                        # Shannon entropy of box density distribution
                        # Normalize densities to probabilities
                        if np.sum(filled_box_densities) > 0:
                            probs = filled_box_densities / np.sum(filled_box_densities)
                            entropy = -np.sum(probs * np.log2(probs + 1e-10))
                            lacunarity_dict[f"lacunarity_entropy_r{box_size}"] = entropy
                        else:
                            lacunarity_dict[f"lacunarity_entropy_r{box_size}"] = 0
                    else:
                        # No filled boxes
                        lacunarity_dict[f"lacunarity_r{box_size}"] = 0
                        lacunarity_dict[f"lacunarity_cv_r{box_size}"] = 0
                        lacunarity_dict[f"lacunarity_entropy_r{box_size}"] = 0
                else:
                    # Empty image or invalid box size
                    lacunarity_dict[f"lacunarity_r{box_size}"] = 0
                    lacunarity_dict[f"lacunarity_all_r{box_size}"] = 0
                    lacunarity_dict[f"lacunarity_gap_ratio_r{box_size}"] = 1 if box_size > 0 else 0
                    lacunarity_dict[f"lacunarity_cv_r{box_size}"] = 0
                    lacunarity_dict[f"lacunarity_entropy_r{box_size}"] = 0
                
            except Exception as e:
                # Log error and set default values for this box size
                print(f"Error calculating lacunarity measures for box size {box_size}: {e}")
                lacunarity_dict[f"lacunarity_r{box_size}"] = 0
                lacunarity_dict[f"lacunarity_all_r{box_size}"] = 0
                lacunarity_dict[f"lacunarity_gap_ratio_r{box_size}"] = 0
                lacunarity_dict[f"lacunarity_cv_r{box_size}"] = 0
                lacunarity_dict[f"lacunarity_entropy_r{box_size}"] = 0
        
        return lacunarity_dict
    
    def analyze_image(self, image_path):
        """
        Analyze an image and return combined fractal and lacunarity measures.
        
        Parameters:
        -----------
        image_path : str or Path
            Path to the image file.
            
        Returns:
        --------
        features : dict
            Dictionary containing all fractal and lacunarity measures.
        """
        # Preprocess the image
        binary_image = self.preprocess_image(image_path)
        
        # Calculate fractal measures
        fractal_dict = self.calculate_fractal_measures(binary_image)
        
        # Calculate lacunarity measures
        lacunarity_dict = self.calculate_lacunarity_measures(binary_image)
        
        # Combine both dictionaries
        features = {**fractal_dict, **lacunarity_dict}
        
        return features


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import os
    
    # Initialize analyzer with custom box sizes
    analyzer = FractalAnalyzer(box_sizes=[1, 2, 3, 4, 5])
    
    # Analyze a single image
    image_path = "sample_image.jpg"  # Replace with your image path
    if os.path.exists(image_path):
        features = analyzer.analyze_image(image_path)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        print(df.head())
        
        # Show available features
        print("\nAvailable Features:")
        for key in features.keys():
            print(f"- {key}")
    else:
        print(f"Sample image not found: {image_path}")