import cv2
import numpy as np
import pyautogui
import pytesseract
import time
import pandas as pd
from datetime import datetime
import os
import json
import keyboard
from PIL import Image

class DarkAndDarkerGameClientScraper:
    def __init__(self, output_folder="price_data", debug_folder="debug_images"):
        self.output_folder = output_folder
        self.debug_folder = debug_folder
        
        # Configure pytesseract path - adjust this to your installation
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Create output folders if they don't exist
        for folder in [output_folder, debug_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            
        # Define regions of interest for the marketplace UI
        # These will need to be calibrated to your screen resolution
        self.regions = {
            "item_name": (300, 200, 400, 30),     # (x, y, width, height)
            "item_price": (700, 200, 100, 30),    # These are examples and will need adjustment
            "item_list": (200, 250, 600, 400),
            "next_page_button": (700, 650, 100, 50)
        }
        
        # Item category locations (for clicking)
        self.categories = {
            "weapons": (100, 100),
            "armor": (100, 150),
            "consumables": (100, 200),
            # Add more categories as needed
        }
        
        # Known items for validation (update with actual game items)
        self.known_items = [
            "leather armor", "longsword", "greataxe", "healing potion", 
            "plate armor", "wizard staff", "bow", "dagger", "cloak",
            "mage robe", "shield", "battleaxe", "health potion"
        ]
        
        # OCR confidence threshold
        self.confidence_threshold = 60  # Percentage
        
    def capture_screen_region(self, region):
        """Capture a specific region of the screen"""
        x, y, width, height = region
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot)
    
    def extract_text_from_image(self, image, is_numeric=False, debug_name=None):
        """Extract text from image using OCR with enhanced preprocessing"""
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply noise reduction
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Save debug image if requested
        if debug_name:
            debug_path = os.path.join(self.debug_folder, f"{debug_name}_original.png")
            cv2.imwrite(debug_path, image)
            
            debug_path = os.path.join(self.debug_folder, f"{debug_name}_processed.png")
            cv2.imwrite(debug_path, thresh)
        
        # OCR configuration
        config = '--psm 7 --oem 3'
        if is_numeric:
            config += ' -c tessedit_char_whitelist="0123456789.,"'
        
        # Get detailed OCR data including confidence levels
        ocr_data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
        
        # Extract text and confidence from OCR results
        texts = []
        confidences = []
        
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > self.confidence_threshold:
                if ocr_data['text'][i].strip():
                    texts.append(ocr_data['text'][i])
                    confidences.append(int(ocr_data['conf'][i]))
        
        # Combine all detected text
        if texts:
            text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        else:
            text = ""
            avg_confidence = 0
        
        return text, avg_confidence
    
    def validate_item_name(self, name):
        """Validate item name against known items and apply corrections"""
        name = name.lower().strip()
        
        # Similarity score function (simple implementation)
        def similarity(s1, s2):
            # Calculate Levenshtein distance
            m, n = len(s1), len(s2)
            if m == 0: return n
            if n == 0: return m
            
            dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
            
            for i in range(m+1):
                dp[i][0] = i
            for j in range(n+1):
                dp[0][j] = j
                
            for i in range(1, m+1):
                for j in range(1, n+1):
                    cost = 0 if s1[i-1] == s2[j-1] else 1
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # deletion
                        dp[i][j-1] + 1,      # insertion
                        dp[i-1][j-1] + cost  # substitution
                    )
            
            # Return similarity score (0 to 1)
            max_len = max(m, n)
            return 1 - (dp[m][n] / max_len) if max_len > 0 else 1
        
        # Find the best match from known items
        best_match = None
        best_score = 0.7  # Minimum similarity threshold
        
        for item in self.known_items:
            score = similarity(name, item)
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_match:
            print(f"OCR read '{name}' - corrected to '{best_match}' (similarity: {best_score:.2f})")
            return best_match, best_score
        
        print(f"Warning: Unrecognized item '{name}' - no close match found")
        return name, 0
    
    def test_ocr_accuracy(self):
        """Interactive tool to test OCR accuracy on specific screen regions"""
        print("OCR Testing Tool")
        print("--------------")
        print("This will help you test if OCR is correctly reading text from the screen.")
        print("Press 'C' to capture the current screen region, or 'Q' to quit.")
        
        while True:
            key = keyboard.read_key()
            
            if key == 'q':
                break
                
            if key == 'c':
                # Get current mouse position
                pos = pyautogui.position()
                
                # Capture region around mouse position
                region = (pos.x - 200, pos.y - 15, 400, 30)  # Adjust as needed
                image = self.capture_screen_region(region)
                
                # Save the captured image
                timestamp = datetime.now().strftime("%H%M%S")
                debug_name = f"test_capture_{timestamp}"
                
                # Extract text with OCR
                text, confidence = self.extract_text_from_image(
                    image, is_numeric=False, debug_name=debug_name
                )
                
                # Validate against known items
                if text:
                    corrected_text, similarity = self.validate_item_name(text)
                    
                    print("\nOCR Test Results:")
                    print(f"Raw text: '{text}' (confidence: {confidence:.1f}%)")
                    print(f"Corrected: '{corrected_text}' (similarity: {similarity:.2f})")
                    print(f"Debug images saved to {self.debug_folder}/{debug_name}_*.png")
                else:
                    print("\nNo text detected in the captured region.")
                
                # Wait before next capture
                time.sleep(1)
    
    def navigate_to_marketplace(self):
        """Navigate to the marketplace in the game"""
        print("Please manually navigate to the marketplace in the game.")
        print("Position the marketplace window where item names and prices are visible.")
        print("Press 'S' when ready to start scanning, or 'T' to enter testing mode.")
        
        key = keyboard.read_key()
        if key == 't':
            print("Entering OCR testing mode...")
            self.test_ocr_accuracy()
            return False
            
        print("Starting marketplace scan...")
        time.sleep(1)  # Short delay to prepare
        return True
    
    def click_category(self, category_name):
        """Click on a category in the marketplace"""
        if category_name in self.categories:
            x, y = self.categories[category_name]
            pyautogui.click(x, y)
            time.sleep(1)  # Wait for the category to load
            return True
        return False
    
    def detect_items_in_view(self):
        """Detect and extract information for all items currently visible"""
        items = []
        
        # Capture the item list region
        item_list_img = self.capture_screen_region(self.regions["item_list"])
        
        # For demonstration - in a real implementation, you would:
        # 1. Use image processing to detect individual item rows
        # 2. For each row, extract the name and price regions
        # 3. Apply OCR to each region
        
        # Simplified approach - scan predefined rows
        row_height = 40  # Estimated height of each item row
        num_rows = self.regions["item_list"][3] // row_height
        
        for i in range(num_rows):
            # Calculate the y-offset for this row
            y_offset = i * row_height
            
            # Extract name region for this row
            name_region = (
                self.regions["item_name"][0],
                self.regions["item_name"][1] + y_offset,
                self.regions["item_name"][2],
                self.regions["item_name"][3]
            )
            name_img = self.capture_screen_region(name_region)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            debug_name = f"item_{i}_name_{timestamp}"
            item_name, name_confidence = self.extract_text_from_image(name_img, debug_name=debug_name)
            
            # Extract price region for this row
            price_region = (
                self.regions["item_price"][0],
                self.regions["item_price"][1] + y_offset,
                self.regions["item_price"][2],
                self.regions["item_price"][3]
            )
            price_img = self.capture_screen_region(price_region)
            debug_name = f"item_{i}_price_{timestamp}"
            price_text, price_confidence = self.extract_text_from_image(
                price_img, is_numeric=True, debug_name=debug_name
            )
            
            # Validate and correct the item name
            corrected_name, similarity = self.validate_item_name(item_name)
            
            # Process the price text to extract numeric value
            try:
                price_value = float(price_text.replace("G", "").replace(",", "").strip())
                
                # Only add if we successfully extracted a name and price
                if corrected_name and price_value > 0 and similarity > 0:
                    items.append({
                        "name": corrected_name,
                        "original_name": item_name,
                        "name_confidence": name_confidence,
                        "price": price_value,
                        "price_confidence": price_confidence,
                        "similarity_score": similarity,
                        "timestamp": datetime.now().isoformat()
                    })
            except (ValueError, TypeError):
                continue  # Skip this item if price conversion fails
        
        return items
    
    def click_next_page(self):
        """Click the next page button"""
        x, y = self.regions["next_page_button"][0] + self.regions["next_page_button"][2] // 2, \
               self.regions["next_page_button"][1] + self.regions["next_page_button"][3] // 2
        pyautogui.click(x, y)
        time.sleep(1.5)  # Wait for the next page to load
    
    def scan_category(self, category_name, max_pages=5):
        """Scan a specific category for items"""
        if not self.click_category(category_name):
            print(f"Category {category_name} not found in configured categories")
            return []
            
        all_items = []
        
        for page in range(max_pages):
            print(f"Scanning page {page+1} of {category_name}...")
            
            # Detect items currently in view
            items = self.detect_items_in_view()
            all_items.extend(items)
            
            # Click next page unless we're on the last page
            if page < max_pages - 1:
                self.click_next_page()
                
        return all_items
    
    def run_full_scan(self):
        """Run a full scan of all categories"""
        if not self.navigate_to_marketplace():
            return {}
        
        all_data = {}
        for category_name in self.categories.keys():
            print(f"Scanning category: {category_name}")
            items = self.scan_category(category_name)
            all_data[category_name] = items
            
            # Save category data
            self.save_data(items, f"{category_name.lower().replace(' ', '_')}.csv")
        
        # Save complete dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.output_folder}/full_scan_{timestamp}.json", "w") as f:
            json.dump(all_data, f, indent=2)
            
        print(f"Scan completed. Saved data to {self.output_folder}/")
        return all_data
    
    def save_data(self, items, filename):
        """Save scraped data to CSV file"""
        if not items:
            return
            
        df = pd.DataFrame(items)
        df.to_csv(f"{self.output_folder}/{filename}", index=False)
    
    def add_known_items(self, items_list):
        """Add items to the known items list for validation"""
        for item in items_list:
            if item.lower() not in self.known_items:
                self.known_items.append(item.lower())
        print(f"Added {len(items_list)} items to known items list. Total: {len(self.known_items)}")
    
    def calibrate_ui_regions(self):
        """Interactive tool to calibrate UI regions"""
        print("UI Calibration Tool")
        print("-------------------")
        print("This will help you set up the screen regions for the marketplace UI.")
        print("For each region, you'll need to identify the top-left corner and dimensions.")
        
        for region_name in self.regions:
            print(f"\nCalibrating region: {region_name}")
            print("Move your mouse to the top-left corner and press 'C'")
            keyboard.wait('c')
            start_pos = pyautogui.position()
            
            print("Now move your mouse to the bottom-right corner and press 'C'")
            keyboard.wait('c')
            end_pos = pyautogui.position()
            
            # Calculate and save the region
            x = start_pos.x
            y = start_pos.y
            width = end_pos.x - start_pos.x
            height = end_pos.y - start_pos.y
            
            self.regions[region_name] = (x, y, width, height)
            print(f"Region {region_name} set to: {self.regions[region_name]}")
        
        print("\nCalibration complete. Here are your configured regions:")
        for region_name, dimensions in self.regions.items():
            print(f"{region_name}: {dimensions}")

# Example usage
if __name__ == "__main__":
    # Create scraper instance
    scraper = DarkAndDarkerGameClientScraper()
    
    # Add known Dark and Darker items to improve recognition
    dark_and_darker_items = [
        "Leather Armor", "Plate Armor", "Chain Mail", "Robe", 
        "Longsword", "Shortsword", "Greataxe", "Battleaxe", "Dagger",
        "Crossbow", "Longbow", "Shortbow", "Wizard Staff", "Spellbook",
        "Health Potion", "Mana Potion", "Bandage", "Torch", "Lockpick",
        "Shield", "Buckler", "Helmet", "Boots", "Gloves", "Cloak",
        "Amulet", "Ring", "Wizard Hat", "Adventurer's Pack",
        # Add more actual game items here
    ]
    scraper.add_known_items(dark_and_darker_items)
    
    # Choose what to run
    print("Dark and Darker Market Scanner")
    print("1. Calibrate UI regions")
    print("2. Test OCR accuracy")
    print("3. Run full marketplace scan")
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        scraper.calibrate_ui_regions()
    elif choice == "2":
        scraper.test_ocr_accuracy()
    elif choice == "3":
        scraper.run_full_scan()
    else:
        print("Invalid choice.")