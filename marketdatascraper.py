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

# You'll need to install these dependencies:
# pip install opencv-python numpy pyautogui pytesseract pandas pillow keyboard

class DarkAndDarkerGameClientScraper:
    def __init__(self, output_folder="price_data"):
        self.output_folder = output_folder
        
        # Configure pytesseract path - adjust this to your installation
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
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
        
    def capture_screen_region(self, region):
        """Capture a specific region of the screen"""
        x, y, width, height = region
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot)
    
    def extract_text_from_image(self, image, is_numeric=False):
        """Extract text from image using OCR"""
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to clean up the image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # OCR configuration
        config = '--psm 7'
        if is_numeric:
            config += ' -c tessedit_char_whitelist="0123456789.,"'
            
        # Perform OCR
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        return text
    
    def navigate_to_marketplace(self):
        """Navigate to the marketplace in the game"""
        print("Please manually navigate to the marketplace in the game.")
        print("Position the marketplace window where item names and prices are visible.")
        print("Press 'S' when ready to start scanning.")
        
        keyboard.wait('s')
        print("Starting marketplace scan...")
        time.sleep(1)  # Short delay to prepare
    
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
            item_name = self.extract_text_from_image(name_img)
            
            # Extract price region for this row
            price_region = (
                self.regions["item_price"][0],
                self.regions["item_price"][1] + y_offset,
                self.regions["item_price"][2],
                self.regions["item_price"][3]
            )
            price_img = self.capture_screen_region(price_region)
            price_text = self.extract_text_from_image(price_img, is_numeric=True)
            
            # Process the price text to extract numeric value
            try:
                price_value = float(price_text.replace("G", "").replace(",", "").strip())
                
                # Only add if we successfully extracted a name and price
                if item_name and price_value > 0:
                    items.append({
                        "name": item_name,
                        "price": price_value,
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
        self.navigate_to_marketplace()
        
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
    scraper = DarkAndDarkerGameClientScraper()
    
    # Run calibration first time
    # scraper.calibrate_ui_regions()
    
    # Run a full scan of the marketplace
    data = scraper.run_full_scan()