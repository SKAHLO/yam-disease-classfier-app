#!/usr/bin/env python3
"""
Yam Leaf Disease Dataset Scraper
Comprehensive web scraping tool for collecting yam leaf disease images
for machine learning/object detection models.
"""

import os
import requests
import urllib.parse
from urllib.parse import urlparse
import time
import json
from pathlib import Path
import hashlib
from typing import List, Dict
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YamDiseaseImageScraper:
    def __init__(self, base_dir="dataset"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Disease categories based on research
        self.disease_categories = {
            'healthy': ['healthy yam leaf', 'normal yam plant leaf'],
            'anthracnose': ['yam anthracnose disease', 'yam leaf anthracnose spots', 'colletotrichum yam disease'],
            'leaf_spot': ['yam leaf spot disease', 'yam concentric leaf spot', 'cercospora yam leaf'],
            'leaf_blight': ['yam leaf blight', 'alternaria yam disease', 'yam leaf blight symptoms'],
            'mosaic_virus': ['yam mosaic virus', 'YMV symptoms yam', 'viral disease yam leaves'],
            'mild_mosaic': ['yam mild mosaic virus', 'YMMV yam disease', 'mild mosaic yam leaf'],
            'bacterial_spot': ['bacterial leaf spot yam', 'bacterial disease yam leaf', 'xanthomonas yam'],
            'bacilliform_virus': ['dioscorea bacilliform virus', 'DBV yam disease']
        }
        
        # Image sources
        self.search_engines = [
            'https://www.google.com/search?q={}&tbm=isch',
            'https://www.bing.com/images/search?q={}',
        ]
        
    def setup_directories(self):
        """Create organized directory structure for dataset"""
        self.base_dir.mkdir(exist_ok=True)
        
        # Create main category directories
        categories = ['healthy', 'anthracnose', 'leaf_spot', 'leaf_blight', 
                     'mosaic_virus', 'mild_mosaic', 'bacterial_spot', 'bacilliform_virus']
        
        for category in categories:
            category_dir = self.base_dir / category
            category_dir.mkdir(exist_ok=True)
            
        # Create metadata directory
        (self.base_dir / 'metadata').mkdir(exist_ok=True)
        
        logger.info(f"Dataset directories created in {self.base_dir}")
        
    def setup_selenium_driver(self):
        """Setup Chrome driver for dynamic content scraping"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            logger.info("Please install ChromeDriver and ensure it's in PATH")
            return None
    
    def get_google_images(self, query: str, max_images: int = 100) -> List[str]:
        """Scrape image URLs from Google Images"""
        driver = self.setup_selenium_driver()
        if not driver:
            return []
            
        image_urls = []
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
        
        try:
            driver.get(search_url)
            time.sleep(2)
            
            # Scroll to load more images
            for i in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Extract image URLs
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            for img in img_elements[:max_images]:
                src = img.get_attribute("src")
                if src and src.startswith(('http', 'data:')):
                    image_urls.append(src)
                    
        except Exception as e:
            logger.error(f"Error scraping Google Images for '{query}': {e}")
        finally:
            driver.quit()
            
        return list(set(image_urls))  # Remove duplicates
    
    def download_image(self, url: str, filepath: Path) -> bool:
        """Download an image from URL"""
        try:
            if url.startswith('data:'):
                # Handle base64 encoded images
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                return True
            else:
                # Handle regular URLs
                response = self.session.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                # Check if it's actually an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    return False
                    
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                return True
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def validate_image(self, filepath: Path) -> bool:
        """Validate downloaded image"""
        try:
            if filepath.stat().st_size < 1024:  # Too small
                return False
                
            # Check if it's a valid image format
            with open(filepath, 'rb') as f:
                header = f.read(16)
                
            # Common image format headers
            image_signatures = [
                b'\xff\xd8\xff',  # JPEG
                b'\x89PNG\r\n\x1a\n',  # PNG
                b'GIF87a',  # GIF
                b'GIF89a',  # GIF
                b'RIFF',  # WebP (partial)
            ]
            
            return any(header.startswith(sig) for sig in image_signatures)
            
        except Exception:
            return False
    
    def scrape_category(self, category: str, max_images_per_query: int = 50):
        """Scrape images for a specific disease category"""
        logger.info(f"Scraping images for category: {category}")
        
        category_dir = self.base_dir / category
        queries = self.disease_categories.get(category, [])
        
        downloaded_count = 0
        metadata = []
        
        for query in queries:
            logger.info(f"Searching for: {query}")
            
            # Get image URLs
            image_urls = self.get_google_images(query, max_images_per_query)
            logger.info(f"Found {len(image_urls)} image URLs for '{query}'")
            
            # Download images
            for i, url in enumerate(image_urls):
                try:
                    # Create unique filename
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"{category}_{query.replace(' ', '_')}_{i}_{url_hash}.jpg"
                    filepath = category_dir / filename
                    
                    # Skip if already exists
                    if filepath.exists():
                        continue
                    
                    # Download and validate
                    if self.download_image(url, filepath):
                        if self.validate_image(filepath):
                            downloaded_count += 1
                            metadata.append({
                                'filename': filename,
                                'category': category,
                                'query': query,
                                'url': url,
                                'download_time': time.time()
                            })
                            logger.info(f"Downloaded: {filename}")
                        else:
                            filepath.unlink()  # Delete invalid image
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i} for '{query}': {e}")
                    continue
        
        # Save metadata
        metadata_file = self.base_dir / 'metadata' / f'{category}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Downloaded {downloaded_count} images for {category}")
        return downloaded_count
    
    def scrape_all_categories(self, max_images_per_query: int = 50):
        """Scrape images for all disease categories"""
        logger.info("Starting comprehensive yam disease dataset scraping")
        
        total_downloaded = 0
        results = {}
        
        for category in self.disease_categories.keys():
            try:
                count = self.scrape_category(category, max_images_per_query)
                results[category] = count
                total_downloaded += count
                
                # Brief pause between categories
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to scrape category {category}: {e}")
                results[category] = 0
        
        # Generate final report
        self.generate_dataset_report(results, total_downloaded)
        
        return results
    
    def generate_dataset_report(self, results: Dict, total_count: int):
        """Generate a comprehensive dataset report"""
        report = {
            'dataset_info': {
                'total_images': total_count,
                'categories': len(results),
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'base_directory': str(self.base_dir)
            },
            'category_breakdown': results,
            'disease_descriptions': {
                'healthy': 'Normal, healthy yam leaves without disease symptoms',
                'anthracnose': 'Dark spots with yellow halos, caused by Colletotrichum species',
                'leaf_spot': 'Concentric rings and brownish spots on leaves',
                'leaf_blight': 'Irregular spots with chlorosis and inward curling',
                'mosaic_virus': 'Mottled patterns and chlorotic patches on leaves',
                'mild_mosaic': 'Mild mottling and mosaic patterns, less severe symptoms',
                'bacterial_spot': 'Angular, water-soaked lesions with yellow halos',
                'bacilliform_virus': 'Generally mild symptoms, may show chlorosis or mosaic'
            }
        }
        
        # Save report
        report_file = self.base_dir / 'dataset_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("DATASET SCRAPING COMPLETE")
        logger.info("="*50)
        logger.info(f"Total images downloaded: {total_count}")
        for category, count in results.items():
            logger.info(f"{category}: {count} images")
        logger.info(f"Dataset saved to: {self.base_dir}")
        logger.info(f"Report saved to: {report_file}")

def main():
    """Main execution function"""
    print("Yam Leaf Disease Dataset Scraper")
    print("="*40)
    
    # Initialize scraper
    scraper = YamDiseaseImageScraper("yam_disease_dataset")
    
    # Configuration
    max_images_per_query = 30  # Adjust based on needs
    
    print(f"Target: {max_images_per_query} images per search query")
    print(f"Disease categories: {len(scraper.disease_categories)}")
    print("\nStarting scraping process...")
    
    # Start scraping
    results = scraper.scrape_all_categories(max_images_per_query)
    
    print(f"\nScraping completed successfully!")
    print(f"Check the 'yam_disease_dataset' directory for your images.")

if __name__ == "__main__":
    main()
