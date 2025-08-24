import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class GearInspector:
    def __init__(self, min_defect_area=50, rotation_step=10, tolerance=5):
        self.min_defect_area = min_defect_area
        self.rotation_step = rotation_step
        self.tolerance = tolerance
    
    def load_and_preprocess(self, image_path):
        """Load image, convert to grayscale, and apply binary threshold"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def normalize_scale(self, ideal_img, sample_img):
        """Scale sample to match ideal gear size and dimensions"""
        ideal_height, ideal_width = ideal_img.shape
        resized_sample = cv2.resize(sample_img, (ideal_width, ideal_height))
        
        ideal_contours, _ = cv2.findContours(ideal_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sample_contours, _ = cv2.findContours(resized_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if ideal_contours and sample_contours:
            ideal_outer = max(ideal_contours, key=cv2.contourArea)
            sample_outer = max(sample_contours, key=cv2.contourArea)
            
            ideal_area = cv2.contourArea(ideal_outer)
            sample_area = cv2.contourArea(sample_outer)
            
            if sample_area > 0:
                scale_factor = np.sqrt(ideal_area / sample_area)
                
                if abs(scale_factor - 1.0) > 0.1:
                    new_width = int(ideal_width * scale_factor)
                    new_height = int(ideal_height * scale_factor)
                    scaled_sample = cv2.resize(resized_sample, (new_width, new_height))
                    
                    if scaled_sample.shape != ideal_img.shape:
                        scaled_sample = cv2.resize(scaled_sample, (ideal_width, ideal_height))
                    
                    return scaled_sample
        
        return resized_sample

    def align_gears(self, ideal_img, sample_img):
        """Align sample gear with ideal gear using rotation"""
        if ideal_img.shape != sample_img.shape:
            sample_img = cv2.resize(sample_img, (ideal_img.shape[1], ideal_img.shape[0]))
        
        best_overlap = 0
        best_angle = 0
        center = (sample_img.shape[1] // 2, sample_img.shape[0] // 2)
        
        for angle in range(0, 360, self.rotation_step):
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(sample_img, rotation_matrix, 
                                    (sample_img.shape[1], sample_img.shape[0]))
            
            if rotated.shape != ideal_img.shape:
                rotated = cv2.resize(rotated, (ideal_img.shape[1], ideal_img.shape[0]))
            
            overlap = cv2.bitwise_and(ideal_img, rotated)
            overlap_score = np.sum(overlap) / 255
            
            if overlap_score > best_overlap:
                best_overlap = overlap_score
                best_angle = angle
        
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        aligned_sample = cv2.warpAffine(sample_img, rotation_matrix, 
                                       (sample_img.shape[1], sample_img.shape[0]))
        
        if aligned_sample.shape != ideal_img.shape:
            aligned_sample = cv2.resize(aligned_sample, (ideal_img.shape[1], ideal_img.shape[0]))
        
        return aligned_sample

    def detect_defects(self, ideal_img, sample_img):
        """Find differences between ideal and sample using XOR"""
        if ideal_img.shape != sample_img.shape:
            sample_img = cv2.resize(sample_img, (ideal_img.shape[1], ideal_img.shape[0]))
        
        diff = cv2.bitwise_xor(ideal_img, sample_img)
        
        kernel = np.ones((3, 3), np.uint8)
        diff_clean = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
        
        return diff_clean

    def classify_defects(self, diff_img):
        """Classify defects as broken or worn teeth based on contour properties"""
        contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        broken_teeth = 0
        worn_teeth = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_defect_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            if area > 500 or aspect_ratio > 2.0:
                broken_teeth += 1
            else:
                worn_teeth += 1
        
        return broken_teeth, worn_teeth

    def analyze_inner_diameter(self, ideal_img, sample_img):
        """Compare inner diameters of ideal and sample gears"""
        def get_inner_radius(img):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[1:]:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        return radius
            
            inverted = cv2.bitwise_not(img)
            contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    return radius
            
            return 0
        
        if sample_img.shape != ideal_img.shape:
            sample_img = cv2.resize(sample_img, (ideal_img.shape[1], ideal_img.shape[0]))
        
        ideal_radius = get_inner_radius(ideal_img)
        sample_radius = get_inner_radius(sample_img)
        
        if abs(sample_radius - ideal_radius) <= self.tolerance:
            status = "identical to ideal"
        elif sample_radius > ideal_radius:
            status = "larger than ideal"
        else:
            status = "smaller than ideal"
        
        return status, ideal_radius, sample_radius

    def inspect_gear(self, ideal_path, sample_path, sample_name):
        """Complete gear inspection pipeline"""
        print(f"\nInspecting {sample_name}...")
        
        ideal_img = self.load_and_preprocess(ideal_path)
        sample_img = self.load_and_preprocess(sample_path)
        
        sample_img = self.normalize_scale(ideal_img, sample_img)
        aligned_sample = self.align_gears(ideal_img, sample_img)
        diff_img = self.detect_defects(ideal_img, aligned_sample)
        
        broken_count, worn_count = self.classify_defects(diff_img)
        diameter_status, ideal_radius, sample_radius = self.analyze_inner_diameter(ideal_img, aligned_sample)
        
        print(f"Gear {sample_name}: {broken_count} broken teeth, {worn_count} worn teeth")
        print(f"Inner diameter is {diameter_status}")
        print(f"Radii - Ideal: {ideal_radius:.1f}px, Sample: {sample_radius:.1f}px")
        
        return {
            'sample': sample_name,
            'broken': broken_count,
            'worn': worn_count,
            'diameter_status': diameter_status,
            'ideal_radius': ideal_radius,
            'sample_radius': sample_radius,
            'diff_image': diff_img,
            'aligned_sample': aligned_sample
        }

    def visualize_inspection(self, ideal_path, sample_path, sample_name):
        """Visualize the inspection process"""
        ideal_img = self.load_and_preprocess(ideal_path)
        sample_img = self.load_and_preprocess(sample_path)
        
        sample_img = self.normalize_scale(ideal_img, sample_img)
        aligned_sample = self.align_gears(ideal_img, sample_img)
        diff_img = self.detect_defects(ideal_img, aligned_sample)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Gear Inspection: {sample_name}', fontsize=16)
        
        axes[0, 0].imshow(ideal_img, cmap='gray')
        axes[0, 0].set_title('Ideal Gear')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(aligned_sample, cmap='gray')
        axes[0, 1].set_title('Aligned Sample')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(diff_img, cmap='gray')
        axes[1, 0].set_title('Detected Defects')
        axes[1, 0].axis('off')
        
        overlay = cv2.cvtColor(aligned_sample, cv2.COLOR_GRAY2RGB)
        overlay[diff_img > 0] = [255, 0, 0]
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Defects Highlighted')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def process_batch(self, ideal_path, sample_files):
        """Process multiple samples and generate report"""
        if not os.path.exists(ideal_path):
            print(f"Error: Ideal gear image not found at {ideal_path}")
            print("Available files in current directory:")
            for file in os.listdir('.'):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  - {file}")
            return []
        
        print(f"Using {ideal_path} as ideal gear reference")
        results = []
        
        for sample_name, sample_path in sample_files.items():
            if os.path.exists(sample_path):
                try:
                    result = self.inspect_gear(ideal_path, sample_path, sample_name)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {sample_name}: {str(e)}")
            else:
                print(f"Warning: {sample_path} not found, skipping {sample_name}")
        
        return results

    def generate_report(self, results):
        """Generate summary report"""
        if not results:
            print("No samples were processed. Please check image file paths.")
            return
        
        print("\n=== INSPECTION SUMMARY ===")
        total_broken = sum(r['broken'] for r in results)
        total_worn = sum(r['worn'] for r in results)
        
        print(f"Total samples processed: {len(results)}")
        print(f"Total broken teeth detected: {total_broken}")
        print(f"Total worn teeth detected: {total_worn}")
        
        diameter_issues = [r for r in results if r['diameter_status'] != 'identical to ideal']
        if diameter_issues:
            print(f"Samples with diameter issues: {len(diameter_issues)}")
            for issue in diameter_issues:
                print(f"  - {issue['sample']}: {issue['diameter_status']}")