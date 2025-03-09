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

class DarkAndDarkerMarketplaceScraper:
    def __init__(self, output_folder="price_data", debug_folder="debug_images"):
        self.output_folder = output_folder
        self.debug_folder = debug_folder
        
        # Configure pytesseract path - adjust this to your installation
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Create output folders if they don't exist
        for folder in [output_folder, debug_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Define column regions based on the marketplace table layout - updated based on screenshot
        # These are initial values that should be calibrated
        self.columns = {
            "item_name": {"x": 53, "width": 265},
            "rarity": {"x": 318, "width": 187},
            "slot": {"x": 505, "width": 191},
            "type": {"x": 696, "width": 189},
            "static_attribute": {"x": 885, "width": 190},
            "random_attribute": {"x": 1075, "width": 187},
            "expires": {"x": 1262, "width": 188},
            "price": {"x": 1450, "width": 116},
            # Optional additional column for quantity/stack (seen in some rows)
            "quantity": {"x": 1556, "width": 150}
        }
        
        # Row parameters
        self.first_row_y = 328  # Y-coordinate of the first item row
        self.row_height = 59   # Height of each item row
        self.num_visible_rows = 10  # Number of visible rows in the marketplace
        
        # Updated rarity colors based on screenshot (BGR format)
        self.rarity_colors = {
            "Poor": (128, 128, 128),      # Gray
            "Common": (255, 255, 255),    # White
            "Uncommon": (0, 255, 0),      # Green (like Ring of Courage)
            "Rare": (255, 120, 0),        # Blue (like Fine Cuirass)
            "Epic": (128, 0, 128),        # Purple (like Ox Pendant)
            "Legendary": (0, 165, 255)    # Orange (like Radiant Cloak)
        }
        
        # Updated known item types based on screenshot
        self.known_types = [
            "Necklace", "Back", "Chest", "Legs", "Primary Weapon", "Hands", 
            "Ring", "Utility", "Gem", "Sword", "Plate", "Cloth", "Leather",
            "Drink", "Consumable", "Shield", "Helmet", "Gloves", "Boots"
        ]
        
        # OCR confidence threshold
        self.confidence_threshold = 40  # Percentage
    
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
        
        # Save debug image if requested
        if debug_name:
            debug_path = os.path.join(self.debug_folder, f"{debug_name}_original.png")
            cv2.imwrite(debug_path, image)
            
            debug_path = os.path.join(self.debug_folder, f"{debug_name}_processed.png")
            cv2.imwrite(debug_path, thresh)
        
        # OCR configuration
        custom_config = '--psm 7 --oem 3'
        if is_numeric:
            # Include comma, period, and 'x' for stack quantities
            custom_config += ' -c tessedit_char_whitelist="0123456789,.x"'
        
        # Perform OCR
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()
        
        return text
    
    def capture_full_table(self, save_debug=True):
        """Capture the entire marketplace table"""
        print("Capturing marketplace table...")
        
        # Capture the full marketplace area including refresh button
        full_height = self.first_row_y + (self.row_height * self.num_visible_rows) + 10
        full_width = 1400  # Based on the screenshot width including buy buttons
        
        full_table = pyautogui.screenshot(region=(0, 0, full_width, full_height))
        
        if save_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_table.save(os.path.join(self.debug_folder, f"full_table_{timestamp}.png"))
        
        return np.array(full_table)
    
    def extract_cell_text(self, row_index, column_name, full_table=None):
        """Extract text from a specific cell in the marketplace table"""
        # Calculate cell coordinates
        x = self.columns[column_name]["x"]
        width = self.columns[column_name]["width"]
        y = self.first_row_y + (row_index * self.row_height)
        height = self.row_height - 5  # Slight reduction to avoid overlap
        
        # Extract the cell region
        if full_table is not None:
            cell_image = full_table[y:y+height, x:x+width]
        else:
            cell_image = self.capture_screen_region((x, y, width, height))
        
        # Debug info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_name = f"row{row_index}_{column_name}_{timestamp}"
        
        # Extract text with OCR
        is_numeric = column_name in ["price", "quantity"]
        text = self.extract_text_from_image(cell_image, is_numeric, debug_name)
        
        return text, cell_image
    
    def detect_rarity_from_color(self, image):
        """Detect item rarity based on the text color in the image"""
        # Extract a representative color from the text (non-black pixels)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Use mask to ignore background
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # If no meaningful pixels, return None
        if cv2.countNonZero(mask) == 0:
            return None
        
        # Get average color of text pixels
        mean_color = cv2.mean(image, mask=mask)[:3]  # Get BGR values
        
        # Find closest rarity by color
        closest_rarity = None
        min_distance = float('inf')
        
        for rarity, color in self.rarity_colors.items():
            # Calculate color distance (simple Euclidean)
            distance = np.sqrt(sum([(a - b) ** 2 for a, b in zip(mean_color, color)]))
            
            if distance < min_distance:
                min_distance = distance
                closest_rarity = rarity
        
        return closest_rarity
    
    def extract_price_value(self, price_text):
        """Extract numeric price value from price text"""
        # Handle the gold coin symbol and numeric values
        # Example: "320" -> 320, "3,500" -> 3500
        if not price_text:
            return None
            
        # Remove non-numeric characters except commas and decimals
        clean_text = ''.join(c for c in price_text if c.isdigit() or c == '.' or c == ',')
        clean_text = clean_text.replace(',', '')
        
        try:
            return float(clean_text)
        except ValueError:
            return None
    
    def extract_quantity(self, quantity_text):
        """Extract quantity information from text (e.g., '37 x 3')"""
        if not quantity_text or 'x' not in quantity_text.lower():
            return None, None
            
        parts = quantity_text.lower().split('x')
        if len(parts) != 2:
            return None, None
            
        try:
            unit_price = float(''.join(c for c in parts[0] if c.isdigit() or c == '.' or c == ',').replace(',', ''))
            quantity = int(''.join(c for c in parts[1] if c.isdigit()))
            return unit_price, quantity
        except ValueError:
            return None, None
    
    def scrape_marketplace_items(self, save_images=True):
        """Scrape all visible items from the marketplace"""
        print("Scraping marketplace items...")
        
        # Capture the full table once for efficiency
        full_table = self.capture_full_table(save_debug=save_images)
        
        items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for row_index in range(self.num_visible_rows):
            try:
                # Check if row is empty by looking at item name
                item_name_text, item_name_img = self.extract_cell_text(row_index, "item_name", full_table)
                
                if not item_name_text:
                    continue  # Skip empty rows
                
                # Extract data from each column
                rarity_text, rarity_img = self.extract_cell_text(row_index, "rarity", full_table)
                slot_text, _ = self.extract_cell_text(row_index, "slot", full_table)
                type_text, _ = self.extract_cell_text(row_index, "type", full_table)
                static_attr_text, _ = self.extract_cell_text(row_index, "static_attribute", full_table)
                random_attr_text, _ = self.extract_cell_text(row_index, "random_attribute", full_table)
                expires_text, _ = self.extract_cell_text(row_index, "expires", full_table)
                price_text, _ = self.extract_cell_text(row_index, "price", full_table)
                quantity_text, _ = self.extract_cell_text(row_index, "quantity", full_table)
                
                # Process price (remove gold coin symbol, parse number)
                price_value = self.extract_price_value(price_text)
                
                # Process quantity if applicable
                unit_price, quantity = self.extract_quantity(quantity_text)
                
                # Detect rarity from color if OCR failed
                if not rarity_text or rarity_text not in self.rarity_colors:
                    detected_rarity = self.detect_rarity_from_color(item_name_img)
                    if detected_rarity:
                        rarity_text = detected_rarity
                
                # Create item record
                item = {
                    "item_name": item_name_text,
                    "rarity": rarity_text,
                    "slot": slot_text,
                    "type": type_text,
                    "static_attribute": static_attr_text,
                    "random_attribute": random_attr_text,
                    "expires": expires_text,
                    "price": price_value,
                    "price_text": price_text,
                    "is_stack": bool(unit_price and quantity),
                    "unit_price": unit_price,
                    "quantity": quantity,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"Found item: {item_name_text} - {rarity_text} - {price_text}")
                items.append(item)
                
            except Exception as e:
                print(f"Error processing row {row_index}: {e}")
        
        return items
    
    def click_refresh_button(self):
        """Click the refresh button in the marketplace"""
        # Refresh button location from the screenshot
        refresh_x = 1791
        refresh_y = 30
        
        pyautogui.click(refresh_x, refresh_y)
        print("Clicked refresh button")
        time.sleep(2)  # Wait for refresh
    
    def run_marketplace_scan(self, num_refreshes=5, delay_between_refreshes=3):
        """Run a full scan of the marketplace with multiple refreshes"""
        all_items = []
        
        print(f"Starting marketplace scan with {num_refreshes} refreshes...")
        
        for i in range(num_refreshes):
            print(f"\nScan {i+1}/{num_refreshes}")
            
            # Scrape current page
            items = self.scrape_marketplace_items()
            all_items.extend(items)
            
            # Save current batch
            self.save_data(items, f"marketplace_scan_{i+1}.csv")
            
            # Refresh the marketplace
            if i < num_refreshes - 1:  # Don't refresh after the last scan
                self.click_refresh_button()
                time.sleep(delay_between_refreshes)
        
        # Save all data
        self.save_data(all_items, "marketplace_complete.csv")
        
        # Create a summary
        self.create_market_summary(all_items)
        
        print(f"\nScan completed. Scraped {len(all_items)} items.")
        print(f"Data saved to {self.output_folder}/")
        
        return all_items
    
    def create_market_summary(self, items):
        """Create a summary of marketplace data"""
        if not items:
            return
            
        df = pd.DataFrame(items)
        
        # Summary by item type
        type_summary = df.groupby('type').agg({
            'item_name': 'count',
            'price': ['mean', 'min', 'max']
        }).reset_index()
        
        # Summary by rarity
        rarity_summary = df.groupby('rarity').agg({
            'item_name': 'count',
            'price': ['mean', 'min', 'max']
        }).reset_index()
        
        # Save summaries
        type_summary.to_csv(f"{self.output_folder}/summary_by_type.csv")
        rarity_summary.to_csv(f"{self.output_folder}/summary_by_rarity.csv")
        
        print("\nMarket Summary:")
        print(f"Total Items: {len(items)}")
        print("\nBy Rarity:")
        for _, row in rarity_summary.iterrows():
            if pd.notna(row[('price', 'mean')]):
                print(f"{row['rarity']}: {row[('item_name', 'count')]} items, Avg: {row[('price', 'mean')]:.1f}g")
            else:
                print(f"{row['rarity']}: {row[('item_name', 'count')]} items, Avg: N/A")
    
    def save_data(self, items, filename):
        """Save scraped data to CSV file"""
        if not items:
            return
            
        df = pd.DataFrame(items)
        df.to_csv(f"{self.output_folder}/{filename}", index=False)
    
    def wait_for_trigger_key(self, key='space'):
        """Wait for a specific key to be pressed before continuing"""
        print(f"Press '{key}' when you are ready...")
        keyboard.wait(key)
        print(f"'{key}' pressed. Continuing...")
        # Add a small delay to account for window focus change
        time.sleep(0.5)
    
    def capture_test_screenshot(self, wait_for_key=True):
        """Capture a test screenshot with an option to wait for a key press"""
        if wait_for_key:
            print("Get ready to switch to your game window.")
            self.wait_for_trigger_key('space')
            
        full_table = self.capture_full_table(save_debug=True)
        print("Test image saved to debug folder")
        return full_table
    
    def calibrate_table_layout(self, wait_for_key=True):
        """Interactive tool to calibrate the marketplace table layout"""
        print("Table Layout Calibration Tool")
        print("----------------------------")
        print("This will help you calibrate the column positions.")
        
        if wait_for_key:
            print("Get ready to switch to your game window.")
            self.wait_for_trigger_key('space')
        
        print("\nFirst, let's set the position of the first row.")
        print("\nMove your mouse to the top of the first item row and press 'C'")
        keyboard.wait('c')
        first_row_y = pyautogui.position().y
        self.first_row_y = first_row_y
        print(f"First row Y position set to: {first_row_y}")
        
        print("\nNow let's calibrate each column. For each column, move your cursor to:")
        print("1. The left edge of the column and press 'L'")
        print("2. The right edge of the column and press 'R'")
        
        for column_name in self.columns.keys():
            print(f"\nCalibrating column: {column_name}")
            
            print("Move to left edge and press 'L'")
            keyboard.wait('l')
            left_x = pyautogui.position().x
            
            print("Move to right edge and press 'R'")
            keyboard.wait('r')
            right_x = pyautogui.position().x
            
            self.columns[column_name]["x"] = left_x
            self.columns[column_name]["width"] = right_x - left_x
            
            print(f"Column {column_name} set to: x={left_x}, width={right_x-left_x}")
        
        print("\nFinally, let's measure the row height.")
        print("Move to the top of the first row and press 'T'")
        keyboard.wait('t')
        top_y = pyautogui.position().y
        
        print("Move to the top of the second row and press 'B'")
        keyboard.wait('b')
        bottom_y = pyautogui.position().y
        
        self.row_height = bottom_y - top_y
        print(f"Row height set to: {self.row_height}")
        
        # Set the number of visible rows
        print("\nHow many rows are visible in the marketplace? (Default is 10)")
        try:
            num_rows = int(input("Enter number of rows: "))
            if num_rows > 0:
                self.num_visible_rows = num_rows
        except ValueError:
            print("Using default value of 10 rows")
        
        print("\nCalibration complete. Here are your configured settings:")
        print(f"First row Y: {self.first_row_y}")
        print(f"Row height: {self.row_height}")
        print(f"Number of visible rows: {self.num_visible_rows}")
        print("Columns:")
        for column_name, dimensions in self.columns.items():
            print(f"  {column_name}: x={dimensions['x']}, width={dimensions['width']}")

# Example usage
if __name__ == "__main__":
    scraper = DarkAndDarkerMarketplaceScraper()
    
    print("Dark and Darker Marketplace Scanner")
    print("1. Calibrate table layout")
    print("2. Capture test image")
    print("3. Run marketplace scan")
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        scraper.calibrate_table_layout(wait_for_key=True)
    elif choice == "2":
        scraper.capture_test_screenshot(wait_for_key=True)
    elif choice == "3":
        num_refreshes = int(input("Enter number of refreshes (1-10): "))
        print("Get ready to switch to your game window.")
        scraper.wait_for_trigger_key('space')
        scraper.run_marketplace_scan(num_refreshes=num_refreshes)
    else:
        print("Invalid choice.")