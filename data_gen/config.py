

"""
Configuration constants for the Spanish OCR Dataset Generator.

This module contains all constant variables used throughout the dataset generation process.
"""

ROOT = "" # adjust according to where you placed this folder, this path should be the absolute path

DATASET_DIR = "data/ocr_data_spanish" # where to store the dataset
IMAGE_TEMPLATES_DIR = "data/old_page_templates" # Directory for background image template, located in ./data/old_page_templates/ , the code will catch all image files from the directory
FONTS_DIR = "data/old_fonts/Junicode_2.211/Junicode/TTF" # Directory for fonts, located in ./data/old_fonts/Junicode_2.211/Junicode/TTF , the code will catch all font files from the directory
CORPUS_DIR = "data/spanish_corpora" # Directory for corpora, located in ./data/spanish_corpora , the code will catch all text files from the directory
MAX_WORKERS = 6 # Number of cpu cores you want to use for dataset production


# Text processing parameters
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
TEXT_SEPARATORS = ["\n\n", "\n", " ", ""]

# Font size parameters
MIN_FONT_SIZE = 15
MAX_FONT_SIZE = 30
DEFAULT_FONT_SIZE = 20

# Margin parameters
MIN_SIDE_MARGIN = 30
MAX_SIDE_MARGIN = 80
MIN_VERTICAL_MARGIN = 40
MAX_VERTICAL_MARGIN = 100

# Column settings
MIN_COLUMN_GAP = 10
MAX_COLUMN_GAP = 60

# Header settings
MIN_HEADER_FONT_SIZE = 40
MAX_HEADER_FONT_SIZE = 120
MIN_HEADER_MARGIN = 25
MAX_HEADER_MARGIN = 50

# Effect probabilities
BLEEDTHROUGH_PROBABILITY = 0.9
INK_BLEMISHES_PROBABILITY = 0.8

# Color settings
SEPIA_COLOR = (90, 70, 50)
INK_COLOR = (10, 5, 0)
INK_OPACITY = 200

configs = {
    "default": {
        "config": {
            "font_size": 20,
            "left_margin": 40,
            "right_margin": 40,
            "top_margin": 40,
            "bottom_margin": 40,
            "alignment": ["left", "justified", "center"],
            "two_column": [True, False],
            "column_gap": 20,
            "header_font_size": 40,
            "header_alignment": "center",
            "header_margin_bottom": 25
        }
    },
    "data/old_page_templates/White/3.jpg": {
        "config": {
            "font_size": 20,
            "left_margin": 40,
            "right_margin": 40,
            "top_margin": 40,
            "bottom_margin": 40,
            "alignment": ["justified"],
            "two_column": True,
            "column_gap": 70,
            "header_alignment": "left",
            "header_margin_bottom": 25
        }
    }
}