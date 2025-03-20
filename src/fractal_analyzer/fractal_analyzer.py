import numpy as np
import cv2
from pathlib import Path
import os
import pandas as pd


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
    
    def calculate_multifractal_spectrum(self, binary_image, q_values=None):
        """
        Calculate multifractal spectrum using q-order moments.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
        q_values : array-like, optional
            Range of q values for multifractal analysis. Default is [-5, -3, -1, 0, 1, 3, 5].
            
        Returns:
        --------
        multifractal_dict : dict
            Dictionary containing multifractal measures.
        """
        if q_values is None:
            q_values = np.array([-5, -3, -1, 0, 1, 3, 5])
        
        multifractal_dict = {}
        
        # For each box size
        for box_size in self.box_sizes:
            # Get box densities
            _, box_densities = self._box_counting(binary_image, box_size)
            flat_densities = box_densities.flatten()
            
            # Remove zeros to prevent division by zero
            positive_densities = flat_densities[flat_densities > 0]
            
            if len(positive_densities) > 0:
                # Normalize densities to probabilities
                total = np.sum(positive_densities)
                probabilities = positive_densities / total if total > 0 else np.zeros_like(positive_densities)
                
                # Calculate partition functions for different q values
                for q in q_values:
                    if q != 1:  # q=1 requires special handling
                        # Partition function Z(q,r)
                        Zq = np.sum(probabilities ** q)
                        multifractal_dict[f"mf_Z_q{q}_r{box_size}"] = Zq
                        
                        # Mass exponent τ(q)
                        if Zq > 0:
                            tau_q = np.log(Zq) / np.log(1.0 / box_size)
                            multifractal_dict[f"mf_tau_q{q}_r{box_size}"] = tau_q
                    else:
                        # For q=1, use entropy formulation
                        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                        multifractal_dict[f"mf_Z_q1_r{box_size}"] = np.exp(entropy)
                        multifractal_dict[f"mf_tau_q1_r{box_size}"] = entropy / np.log(1.0 / box_size)
        
        # Calculate overall multifractal spectrum parameters if enough data
        if len(self.box_sizes) >= 3:
            # For each q, calculate D(q) across scales
            for q in q_values:
                tau_values = [multifractal_dict.get(f"mf_tau_q{q}_r{r}", 0) for r in self.box_sizes]
                log_r_values = [np.log(1.0 / r) for r in self.box_sizes]
                
                # Linear regression to find D(q)
                if len(tau_values) >= 2:
                    try:
                        A = np.vstack([log_r_values, np.ones(len(log_r_values))]).T
                        slope, _ = np.linalg.lstsq(A, tau_values, rcond=None)[0]
                        multifractal_dict[f"mf_D_q{q}"] = slope
                    except:
                        multifractal_dict[f"mf_D_q{q}"] = 0
        
        # Calculate spectrum width and other global metrics
        if all(f"mf_D_q{q}" in multifractal_dict for q in [min(q_values), max(q_values)]):
            # Spectrum width is a key indicator of multifractality strength
            multifractal_dict["mf_spectrum_width"] = multifractal_dict[f"mf_D_q{min(q_values)}"] - multifractal_dict[f"mf_D_q{max(q_values)}"]
            
            # Calculate central tendency and symmetry of the spectrum
            if len(q_values) > 2:
                mid_idx = len(q_values) // 2
                d_values = [multifractal_dict.get(f"mf_D_q{q}", 0) for q in q_values]
                multifractal_dict["mf_spectrum_symmetry"] = np.abs(d_values[0] - d_values[-1]) / multifractal_dict["mf_spectrum_width"] if multifractal_dict["mf_spectrum_width"] > 0 else 0
        
        return multifractal_dict
    
    def calculate_directional_fractal_measures(self, binary_image, angles=None):
        """
        Calculate fractal measures along different directions.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
        angles : array-like, optional
            Angles in degrees to analyze. Default is [0, 45, 90, 135].
            
        Returns:
        --------
        directional_dict : dict
            Dictionary containing directional fractal measures.
        """
        if angles is None:
            angles = [0, 45, 90, 135]
        
        directional_dict = {}
        
        # Ensure binary_image is 8-bit unsigned integer
        if binary_image.dtype != np.uint8:
            binary_image = binary_image.astype(np.uint8)
        
        # For each angle
        for angle in angles:
            try:
                # Create directional kernel for this angle - explicitly set as uint8
                if angle == 0:
                    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
                elif angle == 45:
                    kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
                elif angle == 90:
                    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
                elif angle == 135:
                    kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
                else:
                    continue
                
                # Extract directional components using morphological operations
                directional_component = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                
                # Skip if no pixels in this direction
                if np.sum(directional_component) == 0:
                    directional_dict[f"dir_fractal_dimension_{angle}"] = 0
                    directional_dict[f"dir_density_{angle}"] = 0
                    continue
                
                # Calculate fractal dimension for this direction
                box_counts = []
                box_sizes_used = []
                
                for box_size in self.box_sizes:
                    try:
                        box_count, _ = self._box_counting(directional_component, box_size)
                        if box_count > 0:
                            box_counts.append(np.log(box_count))
                            box_sizes_used.append(np.log(box_size))
                    except Exception as e:
                        print(f"Error in box counting for angle {angle}, box size {box_size}: {e}")
                        continue
                
                # Calculate fractal dimension from log-log slope
                if len(box_counts) >= 2:
                    A = np.vstack([box_sizes_used, np.ones(len(box_sizes_used))]).T
                    slope, _ = np.linalg.lstsq(A, box_counts, rcond=None)[0]
                    directional_dict[f"dir_fractal_dimension_{angle}"] = -slope
                else:
                    directional_dict[f"dir_fractal_dimension_{angle}"] = 0
                
                # Calculate density of directional component
                directional_dict[f"dir_density_{angle}"] = np.sum(directional_component > 0) / (binary_image.shape[0] * binary_image.shape[1])
            
            except cv2.error as e:
                print(f"OpenCV error processing direction {angle}: {e}")
                directional_dict[f"dir_fractal_dimension_{angle}"] = 0
                directional_dict[f"dir_density_{angle}"] = 0
        
        # Calculate anisotropy index (variation in fractal dimension across directions)
        dir_dimensions = [directional_dict.get(f"dir_fractal_dimension_{angle}", 0) for angle in angles]
        if any(dim > 0 for dim in dir_dimensions):
            directional_dict["dir_anisotropy_index"] = np.std(dir_dimensions) / np.mean([d for d in dir_dimensions if d > 0])
            directional_dict["dir_max_dimension"] = np.max(dir_dimensions)
            directional_dict["dir_min_dimension"] = np.min([d for d in dir_dimensions if d > 0]) if any(d > 0 for d in dir_dimensions) else 0
            directional_dict["dir_main_angle"] = angles[np.argmax(dir_dimensions)]
        else:
            directional_dict["dir_anisotropy_index"] = 0
            directional_dict["dir_max_dimension"] = 0
            directional_dict["dir_min_dimension"] = 0
            directional_dict["dir_main_angle"] = 0
        
        return directional_dict
    
    def safe_morphological_operation(self, image, operation, kernel_size, iterations=1):
        """
        Perform morphological operations safely with proper kernel types.
        
        Parameters:
        -----------
        image : ndarray
            Input image
        operation : int
            OpenCV morphological operation (e.g., cv2.MORPH_OPEN)
        kernel_size : int or tuple
            Size of the kernel
        iterations : int
            Number of times to apply the operation
            
        Returns:
        --------
        result : ndarray
            Result of the morphological operation
        """
        try:
            # Ensure input image is 8-bit
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Create proper kernel
            if isinstance(kernel_size, int):
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            else:
                kernel = np.ones(kernel_size, dtype=np.uint8)
            
            # Apply morphological operation
            result = cv2.morphologyEx(image, operation, kernel, iterations=iterations)
            return result
        
        except cv2.error as e:
            print(f"OpenCV error in morphological operation: {e}")
            # Return original image if operation fails
            return image.copy()
    
    def calculate_stroke_features(self, binary_image):
        """
        Calculate features related to stroke properties.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
            
        Returns:
        --------
        stroke_dict : dict
            Dictionary containing stroke-related measures.
        """
        stroke_dict = {}
        
        # Skeletonize the image to get stroke centers
        skeleton = self._skeletonize(binary_image)
        
        # Extract stroke width through distance transform
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Get stroke width along skeleton points
        stroke_widths = []
        skel_points = np.where(skeleton > 0)
        for y, x in zip(skel_points[0], skel_points[1]):
            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
                width = dist_transform[y, x] * 2  # Diameter = 2 * radius
                if width > 0:
                    stroke_widths.append(width)
        
        # Calculate stroke width statistics
        if stroke_widths:
            stroke_dict["stroke_width_mean"] = np.mean(stroke_widths)
            stroke_dict["stroke_width_std"] = np.std(stroke_widths)
            stroke_dict["stroke_width_cv"] = stroke_dict["stroke_width_std"] / stroke_dict["stroke_width_mean"] if stroke_dict["stroke_width_mean"] > 0 else 0
            stroke_dict["stroke_width_min"] = np.min(stroke_widths)
            stroke_dict["stroke_width_max"] = np.max(stroke_widths)
            
            # Percentiles for better distribution characterization
            stroke_dict["stroke_width_25pct"] = np.percentile(stroke_widths, 25)
            stroke_dict["stroke_width_median"] = np.median(stroke_widths)
            stroke_dict["stroke_width_75pct"] = np.percentile(stroke_widths, 75)
            stroke_dict["stroke_width_iqr"] = stroke_dict["stroke_width_75pct"] - stroke_dict["stroke_width_25pct"]
        else:
            # Default values if no skeleton points
            stroke_dict["stroke_width_mean"] = 0
            stroke_dict["stroke_width_std"] = 0
            stroke_dict["stroke_width_cv"] = 0
            stroke_dict["stroke_width_min"] = 0
            stroke_dict["stroke_width_max"] = 0
            stroke_dict["stroke_width_25pct"] = 0
            stroke_dict["stroke_width_median"] = 0
            stroke_dict["stroke_width_75pct"] = 0
            stroke_dict["stroke_width_iqr"] = 0
        
        # Calculate junction analysis
        junctions = self._find_junctions(skeleton)
        stroke_dict["stroke_junction_count"] = len(junctions)
        stroke_dict["stroke_junction_density"] = len(junctions) / np.sum(skeleton > 0) if np.sum(skeleton > 0) > 0 else 0
        
        # Estimate stroke length
        stroke_dict["stroke_total_length"] = np.sum(skeleton > 0)
        
        # Analyze stroke segments (between junctions)
        segments = self._segment_skeleton(skeleton, junctions)
        segment_lengths = [len(seg) for seg in segments]
        
        if segment_lengths:
            stroke_dict["stroke_segment_count"] = len(segment_lengths)
            stroke_dict["stroke_segment_mean_length"] = np.mean(segment_lengths)
            stroke_dict["stroke_segment_std_length"] = np.std(segment_lengths)
            stroke_dict["stroke_segment_max_length"] = np.max(segment_lengths)
            stroke_dict["stroke_segment_min_length"] = np.min(segment_lengths)
        else:
            stroke_dict["stroke_segment_count"] = 0
            stroke_dict["stroke_segment_mean_length"] = 0
            stroke_dict["stroke_segment_std_length"] = 0
            stroke_dict["stroke_segment_max_length"] = 0
            stroke_dict["stroke_segment_min_length"] = 0
        
        return stroke_dict

    def _skeletonize(self, binary_image):
        """Skeletonize binary image using OpenCV morphological operations with proper kernel types."""
        # Create a copy to avoid modifying the original
        img = binary_image.copy()
        
        # Ensure image is 8-bit single-channel
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Initialize skeleton
        skeleton = np.zeros_like(img, dtype=np.uint8)
        
        # Create structuring element - explicitly set as uint8
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        # Keep track of last iteration to prevent infinite loops
        prev_img = None
        max_iterations = 100  # Safety limit
        iterations = 0
        
        while True:
            # Store current image for comparison
            prev_img = img.copy()
            
            try:
                # Erode the image - with error handling
                eroded = cv2.erode(img, kernel)
                
                # Dilate the eroded image
                temp = cv2.dilate(eroded, kernel)
                
                # Subtract to get the boundary
                temp = cv2.subtract(img, temp)
                
                # Add boundary to skeleton
                skeleton = cv2.bitwise_or(skeleton, temp)
                
                # Set eroded image for next iteration
                img = eroded.copy()
                
                # Check if image is empty or unchanged
                if cv2.countNonZero(img) == 0 or np.array_equal(img, prev_img) or iterations > max_iterations:
                    break
                    
                iterations += 1
                
            except cv2.error as e:
                print(f"OpenCV error during skeletonization: {e}")
                # Return what we have so far, or the original if early failure
                return skeleton if cv2.countNonZero(skeleton) > 0 else binary_image
        
        return skeleton

    def _find_junctions(self, skeleton):
        """Find junction points in the skeleton."""
        junctions = []
        
        # Create kernel for neighbor counting
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # Don't count center pixel
        
        # Count neighbors for each pixel
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # Junction points have more than 2 neighbors
        junction_points = np.logical_and(skeleton > 0, neighbor_count >= 3)
        junction_coords = np.where(junction_points)
        
        for y, x in zip(junction_coords[0], junction_coords[1]):
            junctions.append((x, y))
        
        return junctions

    def _segment_skeleton(self, skeleton, junctions):
        """Segment skeleton into parts between junctions."""
        # Create a map of junction points
        junction_map = np.zeros_like(skeleton)
        for x, y in junctions:
            junction_map[y, x] = 1
        
        # Remove junctions from skeleton to get segments
        segments_map = np.logical_and(skeleton > 0, junction_map == 0).astype(np.uint8)
        
        # Label connected components to identify segments
        num_labels, labels = cv2.connectedComponents(segments_map)
        
        # Extract each segment as list of coordinates
        segments = []
        for i in range(1, num_labels):
            segment_coords = np.where(labels == i)
            if len(segment_coords[0]) > 0:
                segment = [(x, y) for y, x in zip(segment_coords[0], segment_coords[1])]
                segments.append(segment)
        
        return segments
    
    def calculate_topological_features(self, binary_image):
        """
        Calculate topological features using persistent homology concepts.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
            
        Returns:
        --------
        topo_dict : dict
            Dictionary containing topological features.
        """
        topo_dict = {}
    
        # Ensure binary_image is 8-bit unsigned integer
        if binary_image.dtype != np.uint8:
            binary_image = binary_image.astype(np.uint8)
        
        # Initialize for persistence diagram calculations
        persistence_birth_death = []
        
        # Calculate connected components at multiple thresholds
        for i, box_size in enumerate(sorted(self.box_sizes)):
            try:
                # Create properly-typed kernel
                kernel = np.ones((box_size, box_size), dtype=np.uint8)
                
                # Dilate the image to simulate different threshold levels
                dilated = cv2.dilate(binary_image, kernel)
                
                # Find connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
                
                # Record basic stats for this box size
                topo_dict[f"topo_components_r{box_size}"] = num_labels - 1  # Subtract background
                
                # Calculate area statistics of components
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
                
                if len(areas) > 0:
                    topo_dict[f"topo_mean_area_r{box_size}"] = np.mean(areas)
                    topo_dict[f"topo_std_area_r{box_size}"] = np.std(areas)
                    topo_dict[f"topo_max_area_r{box_size}"] = np.max(areas)
                else:
                    topo_dict[f"topo_mean_area_r{box_size}"] = 0
                    topo_dict[f"topo_std_area_r{box_size}"] = 0
                    topo_dict[f"topo_max_area_r{box_size}"] = 0
                
                # Tracking for persistence diagram (omitted for brevity)
                # ...
            
            except cv2.error as e:
                print(f"OpenCV error processing box size {box_size}: {e}")
                topo_dict[f"topo_components_r{box_size}"] = 0
                topo_dict[f"topo_mean_area_r{box_size}"] = 0
                topo_dict[f"topo_std_area_r{box_size}"] = 0
                topo_dict[f"topo_max_area_r{box_size}"] = 0
        
        # Calculate Euler characteristic at each scale
        for box_size in self.box_sizes:
            try:
                # Calculate Euler characteristic safely
                kernel = np.ones((box_size, box_size), dtype=np.uint8)
                dilated = cv2.dilate(binary_image, kernel)
                
                # Count connected components
                num_components = cv2.connectedComponents(dilated)[0] - 1
                
                # Count holes (using the inverse image)
                inverted = cv2.bitwise_not(dilated)
                num_holes = cv2.connectedComponents(inverted)[0] - 1
                
                # Euler characteristic
                euler = num_components - num_holes
                topo_dict[f"topo_euler_r{box_size}"] = euler
            except cv2.error as e:
                print(f"OpenCV error calculating Euler characteristic for box size {box_size}: {e}")
                topo_dict[f"topo_euler_r{box_size}"] = 0
        
        # Calculate Betti numbers from existing values
        topo_dict["topo_betti0"] = topo_dict.get("topo_components_r2", 0)
        topo_dict["topo_betti1"] = topo_dict.get("topo_components_r2", 0) - topo_dict.get("topo_euler_r2", 0)
        
        return topo_dict
    
    def calculate_spatial_distribution(self, binary_image):
        """
        Calculate features related to spatial distribution of handwriting.
        
        Parameters:
        -----------
        binary_image : ndarray
            Binary image to analyze.
            
        Returns:
        --------
        spatial_dict : dict
            Dictionary containing spatial distribution features.
        """
        spatial_dict = {}
        h, w = binary_image.shape
        
        # Find foreground pixels
        y_indices, x_indices = np.where(binary_image > 0)
        
        if len(y_indices) == 0:
            # No foreground pixels
            spatial_dict["spatial_centroid_x"] = w / 2
            spatial_dict["spatial_centroid_y"] = h / 2
            spatial_dict["spatial_coverage"] = 0
            spatial_dict["spatial_dispersion"] = 0
            spatial_dict["spatial_horizontal_balance"] = 0.5
            spatial_dict["spatial_vertical_balance"] = 0.5
            return spatial_dict
        
        # Calculate centroid
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        spatial_dict["spatial_centroid_x"] = centroid_x
        spatial_dict["spatial_centroid_y"] = centroid_y
        
        # Calculate normalized centroid position (0-1 range)
        spatial_dict["spatial_centroid_x_norm"] = centroid_x / w
        spatial_dict["spatial_centroid_y_norm"] = centroid_y / h
        
        # Calculate coverage (foreground pixel percentage)
        spatial_dict["spatial_coverage"] = len(y_indices) / (h * w)
        
        # Calculate dispersion (std deviation of pixel distances from centroid)
        distances = np.sqrt((y_indices - centroid_y)**2 + (x_indices - centroid_x)**2)
        spatial_dict["spatial_dispersion"] = np.std(distances) / np.sqrt(h**2 + w**2)  # Normalize by diagonal length
        
        # Calculate horizontal and vertical balance
        # (ratio of pixels left/right or above/below centroid)
        left_count = np.sum(x_indices < centroid_x)
        right_count = np.sum(x_indices >= centroid_x)
        spatial_dict["spatial_horizontal_balance"] = left_count / (left_count + right_count) if (left_count + right_count) > 0 else 0.5
        
        top_count = np.sum(y_indices < centroid_y)
        bottom_count = np.sum(y_indices >= centroid_y)
        spatial_dict["spatial_vertical_balance"] = top_count / (top_count + bottom_count) if (top_count + bottom_count) > 0 else 0.5
        
        # Divide image into 3x3 grid and measure density in each cell
        grid_size = 3
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate grid cell boundaries
                y_min = i * h // grid_size
                y_max = (i + 1) * h // grid_size
                x_min = j * w // grid_size
                x_max = (j + 1) * w // grid_size
                
                # Count pixels in this cell
                cell_count = np.sum((y_indices >= y_min) & (y_indices < y_max) & 
                                (x_indices >= x_min) & (x_indices < x_max))
                
                # Calculate density
                cell_density = cell_count / ((y_max - y_min) * (x_max - x_min))
                spatial_dict[f"spatial_density_grid_{i}_{j}"] = cell_density
        
        # Calculate entropy of spatial distribution across the grid
        grid_densities = [spatial_dict[f"spatial_density_grid_{i}_{j}"] 
                        for i in range(grid_size) for j in range(grid_size)]
        
        if np.sum(grid_densities) > 0:
            probs = np.array(grid_densities) / np.sum(grid_densities)
            spatial_dict["spatial_grid_entropy"] = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            spatial_dict["spatial_grid_entropy"] = 0
        
        # Calculate spatial autocorrelation (Moran's I)
        # This measures how similar nearby grid cells are
        spatial_dict["spatial_autocorrelation"] = self._calculate_morans_i(grid_densities, grid_size)
        
        return spatial_dict

    def _calculate_morans_i(self, values, grid_size):
        """Calculate Moran's I spatial autocorrelation index."""
        n = len(values)
        if n <= 1 or np.std(values) == 0:
            return 0
        
        # Create adjacency matrix for queen's case (all 8 surrounding cells)
        W = np.zeros((n, n))
        for i in range(grid_size):
            for j in range(grid_size):
                idx1 = i * grid_size + j
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            idx2 = ni * grid_size + nj
                            W[idx1, idx2] = 1
        
        # Row standardize the weights
        W_sum = W.sum(axis=1)
        W_sum[W_sum == 0] = 1  # Avoid division by zero
        W = W / W_sum[:, np.newaxis]
        
        # Calculate Moran's I
        z = values - np.mean(values)
        z_norm = z / np.std(values)
        
        num = np.sum(np.outer(z_norm, z_norm) * W)
        den = n
        
        return num / den
    
    def analyze_image(self, image_path):
        """
        Analyze an image and return combined measures from all feature extractors.
        
        Parameters:
        -----------
        image_path : str or Path
            Path to the image file.
            
        Returns:
        --------
        features : dict
            Dictionary containing all extracted features.
        """
        # Preprocess the image
        binary_image = self.preprocess_image(image_path)
        
        # Original features
        fractal_dict = self.calculate_fractal_measures(binary_image)
        lacunarity_dict = self.calculate_lacunarity_measures(binary_image)
        
        # New advanced features
        multifractal_dict = self.calculate_multifractal_spectrum(binary_image)
        directional_dict = self.calculate_directional_fractal_measures(binary_image)
        stroke_dict = self.calculate_stroke_features(binary_image)
        topological_dict = self.calculate_topological_features(binary_image)
        spatial_dict = self.calculate_spatial_distribution(binary_image)
        
        # Combine all dictionaries
        features = {
            **fractal_dict, 
            **lacunarity_dict,
            **multifractal_dict,
            **directional_dict,
            **stroke_dict,
            **topological_dict,
            **spatial_dict
        }
        
        return features


# Example usage
if __name__ == "__main__":
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