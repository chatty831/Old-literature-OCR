"""
Spanish OCR Dataset Generator

This module creates an OCR dataset by transforming modern Spanish text to old Spanish text
and rendering it on historical-looking page templates with various effects.
"""

from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import math
import re
import random
import json
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from data_gen.data_gen import old_to_modern_spanish, modern_to_old_spanish, simulate_old_book
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import constants from config.py
from data_gen.config import (
    ROOT,
    DATASET_DIR,
    IMAGE_TEMPLATES_DIR,
    FONTS_DIR,
    CORPUS_DIR,
    MAX_WORKERS,
    configs
)

IMAGE_TEMPLATES_DIR = os.path.join(ROOT, IMAGE_TEMPLATES_DIR)
FONTS_DIR = os.path.join(ROOT, FONTS_DIR)
CORPUS_DIR = os.path.join(ROOT, CORPUS_DIR)
DATASET_DIR = os.path.join(ROOT, DATASET_DIR)


@lru_cache(maxsize=None)
def get_hyphenation_positions(word):
    """
    Find potential hyphenation points in a word.
    
    Args:
        word: The word to analyze for hyphenation
        
    Returns:
        list: Positions where the word could be hyphenated
    """
    # Simple rule-based hyphenation - can be improved with dictionary-based approach
    # Returns positions where you can insert hyphens
    if len(word) <= 4:  # Don't hyphenate short words
        return []
    
    positions = []
    # Look for natural word breaks like compound words
    compound_word_pattern = r'[a-z][A-Z]'
    matches = list(re.finditer(compound_word_pattern, word))
    for match in matches:
        positions.append(match.start() + 1)
    
    # Traditional hyphenation rules (simplified)
    # 1. Try to break between syllables
    vowels = 'aeiouAEIOU'
    for i in range(2, len(word) - 2):
        # Break after vowel followed by consonant
        if (word[i-1] in vowels and word[i] not in vowels) or \
           (word[i-1] not in vowels and word[i] not in vowels and word[i-1] != word[i]):
            positions.append(i)
    
    # Remove duplicates and sort
    positions = sorted(list(set(positions)))
    
    return positions

# Various caches for different operations
_getbbox_cache = {}
_word_width_cache = {}
_hyphenation_cache = {}
_space_width_cache = {}

def wrap_text_with_hyphenation(text, font, max_width, draw):
    """
    Wrap text with proper hyphenation, with caching for expensive operations.
    
    Args:
        text: Text to wrap
        font: Font to use for width calculations
        max_width: Maximum width in pixels
        draw: ImageDraw object for text measurements
        
    Returns:
        list: Wrapped lines with hyphenation where appropriate
    """
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    # Get space width (cached)
    font_id = id(font)
    space_width = _space_width_cache.get(font_id, None)
    if space_width is None:
        space_width = cached_getbbox(font, " ")[2] - cached_getbbox(font, " ")[0]
        _space_width_cache[font_id] = space_width
    
    for word in words:
        # Calculate width of current word (cached)
        word_width = get_cached_word_width(font, word)
        
        # If adding this word exceeds max width
        if current_width + word_width + (space_width if current_line else 0) > max_width:
            # Try hyphenation if word is long
            if word_width > max_width * 0.3 and len(word) > 5:
                # Get hyphenation positions (cached)
                hyphen_positions = get_cached_hyphenation(word)
                
                if hyphen_positions:
                    # Try each hyphenation point
                    for pos in hyphen_positions:
                        first_part = word[:pos] + "-"
                        first_part_width = get_cached_word_width(font, first_part)
                        
                        # If first part fits
                        if current_width + first_part_width + (space_width if current_line else 0) <= max_width:
                            # Add first part with hyphen to current line
                            if current_line:
                                current_line.append(first_part)
                                lines.append(" ".join(current_line))
                            else:
                                lines.append(first_part)
                                
                            # Start new line with second part
                            second_part = word[pos:]
                            current_line = [second_part]
                            current_width = get_cached_word_width(font, second_part)
                            break
                    else:
                        # If no hyphenation worked, add current line and start new with full word
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]
                        current_width = word_width
                else:
                    # No good hyphenation points, add current line and start new with full word
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
            else:
                # Word not suitable for hyphenation, add current line and start new with full word
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
        else:
            # Word fits on current line
            current_line.append(word)
            current_width += word_width + (space_width if current_line else 0)
    
    # Add last line if not empty
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines


def cached_getbbox(font, text):
    """
    Cached version of font.getbbox to avoid repeated calculations.
    """
    cache_key = (id(font), text)
    bbox = _getbbox_cache.get(cache_key, None)
    if bbox is None:
        bbox = font.getbbox(text)
        _getbbox_cache[cache_key] = bbox
    return bbox


def get_cached_word_width(font, word):
    """
    Get cached word width measurement.
    """
    cache_key = (id(font), word)
    width = _word_width_cache.get(cache_key, None)
    if width is None:
        bbox = cached_getbbox(font, word)
        width = bbox[2] - bbox[0]
        _word_width_cache[cache_key] = width
    return width


def get_cached_hyphenation(word):
    """
    Get cached hyphenation positions for a word.
    """
    hyphen_positions = _hyphenation_cache.get(word, None)
    if hyphen_positions is None:
        hyphen_positions = get_hyphenation_positions(word)
        _hyphenation_cache[word] = hyphen_positions
    return hyphen_positions



def add_bleedthrough_text(image, text_positions, font, bleedthrough_offset=8, 
                         bleedthrough_opacity=0.15, sepia_color=(90, 70, 50)):
    """
    Adds see-through text that appears to be bleeding through from the back of the page,
    positioned above each line of text.
    
    Args:
        image: PIL Image object
        text_positions: List of tuples (line, x_pos, y_pos) containing text and positions
        font: PIL ImageFont object
        bleedthrough_offset: Pixels above the line to position the bleedthrough
        bleedthrough_opacity: Fixed opacity value for all bleedthrough text
        sepia_color: Fixed sepia color for all bleedthrough text
        
    Returns:
        PIL Image with bleedthrough text added
    """
    # Work on a copy of the image
    result = image.copy()
    
    # Convert opacity fraction to 0-255 range for PIL
    opacity = int(255 * bleedthrough_opacity)
    
    # Iterate through each text line
    for line, x_pos, y_pos in text_positions:
        if not line.strip():  # Skip empty lines
            continue
            
        # Calculate the mirrored text position (offset pixels above the actual text)
        mirror_y_pos = y_pos - bleedthrough_offset
        
        # Skip if the position would be off the page
        if mirror_y_pos < 5:
            continue
        
        # Create an image for the text line
        text_bbox = font.getbbox(line)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Create a temporary image for the text
        text_img = Image.new('RGBA', (text_width + 20, text_height * 2), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        
        # Draw the text on this temporary image
        text_draw.text((10, 0), line, font=font, fill=(0, 0, 0, 255))
        
        # Mirror the text image horizontally
        mirrored_text = ImageOps.mirror(text_img)
        
        # Color the mirrored text with the fixed sepia color and opacity
        pixels = mirrored_text.load()
        for i in range(mirrored_text.width):
            for j in range(mirrored_text.height):
                r, g, b, a = pixels[i, j]
                if a > 0:  # If there's any opacity
                    # Apply sepia color with the fixed opacity
                    pixels[i, j] = (sepia_color[0], sepia_color[1], sepia_color[2], 
                                  min(a, opacity))
        
        # Paste the mirrored text onto the page with transparency
        result.paste(mirrored_text, (x_pos, mirror_y_pos), mirrored_text)
    
    return result

def add_ink_blemishes(image, num_blemishes=(3, 8), size_range=(2, 12), 
                     color=(10, 5, 0), opacity=220):
    """
    Adds random ink blemishes to simulate aging on the page with more natural edges.
    
    Args:
        image: PIL Image object
        num_blemishes: Tuple (min, max) number of blemishes to add (reduced)
        size_range: Tuple (min, max) size of blemishes in pixels (reduced)
        color: Fixed dark color for ink blemishes
        opacity: Fixed opacity value (0-255)
        
    Returns:
        PIL Image with ink blemishes added
    """
    # Work on a copy of the image
    result = image.copy()
    draw = ImageDraw.Draw(result, 'RGBA')
    
    # Determine image dimensions
    width, height = image.size
    
    # Determine number of blemishes to add
    num = random.randint(num_blemishes[0], num_blemishes[1])
    
    for _ in range(num):
        # Random position
        x = random.randint(0, width)
        y = random.randint(0, height)
        
        # Random size (reduced from original)
        size = random.randint(size_range[0], size_range[1])
        
        # Fixed dark color and opacity
        rgba_color = color + (opacity,)
        
        # Create ink blemish with more natural irregular shape
        if random.random() < 0.8:  # Irregular shape 80% of the time
            # Create an irregular polygon for the blemish with more points for natural look
            num_points = random.randint(15, 22)  # Increased from 5-10 to 15-22
            points = []
            
            # Add some randomness to the shape
            prev_radius_factor = 1.0
            
            for i in range(num_points):
                angle = (i / num_points) * 2 * math.pi
                # More subtle radius variation for natural look
                radius_factor = prev_radius_factor + random.uniform(-0.2, 0.2)
                radius_factor = max(0.6, min(1.4, radius_factor))  # Constrain variation
                prev_radius_factor = radius_factor
                
                radius = size * radius_factor
                px = x + math.cos(angle) * radius
                py = y + math.sin(angle) * radius
                points.append((px, py))
            
            # Draw the polygon with the fixed dark color
            draw.polygon(points, fill=rgba_color)
            
            # Sometimes add small spikes or drips (less frequently)
            if random.random() < 0.2:  # Reduced from 0.3 to 0.2
                start_point = random.choice(points)
                angle = random.random() * 2 * math.pi
                end_x = start_point[0] + math.cos(angle) * size * 1.2
                end_y = start_point[1] + math.sin(angle) * size * 1.2
                draw.line([start_point, (end_x, end_y)], fill=rgba_color, width=random.randint(1, 2))
        else:
            # Simple ellipse for small spots
            ellipse_width = size * (0.8 + random.random() * 0.4)
            ellipse_height = size * (0.8 + random.random() * 0.4)
            draw.ellipse(
                [x - ellipse_width//2, y - ellipse_height//2, 
                 x + ellipse_width//2, y + ellipse_height//2], 
                fill=rgba_color
            )
    
    # Add a couple of streaks or smudges (reduced number)
    for _ in range(random.randint(0, 2)):  # Reduced from 0-5 to 0-2
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        length = random.randint(10, 40)  # Reduced from 20-100 to 10-40
        angle = random.random() * 2 * math.pi
        end_x = start_x + math.cos(angle) * length
        end_y = start_y + math.sin(angle) * length
        
        # Thinner streak width
        width_streak = random.randint(1, 3)  # Reduced from 1-5 to 1-3
        
        # Use the same dark color but with reduced opacity
        streak_opacity = opacity // 2
        streak_color = color + (streak_opacity,)
        
        draw.line([(start_x, start_y), (end_x, end_y)], fill=streak_color, width=width_streak)
    
    return result

def draw_justified_text(draw, line, font, x_position, y_position, column_width, is_paragraph_end=False, color=(0, 0, 0)):
    """
    Draw text with justified alignment, except for paragraph end lines.
    Returns the position information for text tracking.
    
    Args:
        draw: PIL ImageDraw object
        line: Text line to justify
        font: PIL ImageFont object
        x_position: Starting x position
        y_position: Starting y position
        column_width: Width of the column to justify within
        is_paragraph_end: Whether this is the last line of a paragraph
        color: Color of the text
        
    Returns:
        tuple: The (x_position, y_position) of the text
    """
    # If the line is empty, a paragraph end, or has only one word, draw it left-aligned
    words = line.split()
    if not words or len(words) == 1 or is_paragraph_end:
        draw.text((x_position, y_position), line, font=font, fill=color)
        return (x_position, y_position)
    
    # Calculate the width of all words combined
    word_widths = [font.getbbox(word)[2] - font.getbbox(word)[0] for word in words]
    words_width = sum(word_widths)
    
    # Calculate the space width for this font
    space_width = font.getbbox(" ")[2] - font.getbbox(" ")[0]
    
    # Calculate how much extra space we need to distribute
    total_spaces = len(words) - 1
    regular_spaces_width = total_spaces * space_width
    extra_space = column_width - words_width - regular_spaces_width
    
    # If extra space is negative, text is too long, fall back to left-align
    if extra_space < 0:
        draw.text((x_position, y_position), line, font=font, fill=color)
        return (x_position, y_position)
    
    # Calculate the width of each space, including extra space
    if total_spaces > 0:
        adjusted_space_width = space_width + (extra_space / total_spaces)
    else:
        adjusted_space_width = 0
    
    # Draw each word at its correct position
    current_x = x_position
    for i, word in enumerate(words):
        draw.text((current_x, y_position), word, font=font, fill=color)
        if i < len(words) - 1:  # Don't add space after the last word
            current_x += word_widths[i] + adjusted_space_width
    
    return (x_position, y_position)


def simulate_old_book(text, image_template_path, output_dir, font_path, font_size, 
                     left_margin=50, right_margin=50, top_margin=50, bottom_margin=50, 
                     alignment='left', two_column=False, column_gap=20,
                     header_text=None, header_font_size=None, header_alignment='center',
                     header_margin_bottom=20, enable_bleedthrough=False, enable_ink_blemishes=False,
                     bleedthrough_offset=8, bleedthrough_opacity=0.15, sepia_color=(90, 70, 50),
                     num_blemishes=(3, 8), size_range=(2, 12), color=(10, 5, 0), opacity=220):
    """
    Simulates printing text on old book pages.
    
    Args:
        text (str): The text to be printed on the pages.
        image_template_path (str): Path to the template image.
        output_dir (str): Directory to save the output images.
        font_path (str): Path to the TTF font file.
        font_size (int): Size of the font.
        left_margin (int): Margin from the left edge of the page in pixels.
        right_margin (int): Margin from the right edge of the page in pixels.
        top_margin (int): Margin from the top edge of the page in pixels.
        bottom_margin (int): Margin from the bottom edge of the page in pixels.
        alignment (str, optional): Text alignment ('left', 'center', 'right', 'justified'). 
                                   Defaults to 'left'.
        two_column (bool, optional): Whether to use a two-column layout. Defaults to False.
        column_gap (int, optional): Gap between columns in pixels when using two-column layout.
        header_text (str, optional): Text to display as header. Defaults to None.
        header_font_size (int, optional): Font size for header. Defaults to None (1.5x main font).
        header_alignment (str, optional): Alignment for header text. Defaults to 'center'.
        header_margin_bottom (int, optional): Space between header and main text. Defaults to 20px.
        enable_bleedthrough (bool, optional): Enable see-through text effect. Defaults to False.
        enable_ink_blemishes (bool, optional): Enable random ink blemishes. Defaults to False.
    
    Returns:
        tuple: (number of pages generated, dict mapping page numbers to text content)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to track which text appears on which page
    page_text_mapping = {}
    
    # Load the main font
    font = ImageFont.truetype(font_path, font_size)
    
    # Configure header font if header text is provided
    header_font = None
    header_height = 0
    if header_text:
        # If header font size not specified, make it 1.5x the main font size
        if header_font_size is None:
            header_font_size = int(font_size * 1.5)
        header_font = ImageFont.truetype(font_path, header_font_size)
        
        # Calculate header height with padding
        header_bbox = header_font.getbbox(header_text)
        header_height = (header_bbox[3] - header_bbox[1]) + header_margin_bottom
    
    # Calculate line height for main text (with some spacing)
    line_height = int(font_size * 1.5)
    
    # Get template dimensions
    with Image.open(image_template_path) as template:
        template_width, template_height = template.size
    
    # Determine if the page is square-ish
    is_square = 0.8 < template_width/template_height < 1.2
    print(f"Template dimensions: {template_width}x{template_height}, Is square-ish: {is_square}")
    
    # Calculate column width based on layout
    if two_column:
        column_width = (template_width - left_margin - right_margin - column_gap) // 2
    else:
        column_width = template_width - left_margin - right_margin
    
    # Process text - first split by paragraphs (\n\n)
    paragraphs = text.split('\n\n')
    all_lines = []
    paragraph_end_lines = set()  # Track the indices of lines that end paragraphs
    
    # Create a dummy draw object for text measurements
    dummy_img = Image.new('RGB', (10, 10), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    # Process each paragraph - paragraphs are separated by double newlines (\n\n)
    current_line_idx = 0
    # Split text by double newlines to get actual paragraphs
    paragraphs = text.split('\n\n')

    for paragraph in tqdm(paragraphs, total=len(paragraphs), desc="Processing paragraphs"):
        # Process single newlines within a paragraph as line breaks, not paragraph breaks
        paragraph_lines = paragraph.split('\n')
        
        for p_idx, p_line in enumerate(paragraph_lines):
            # Use our hyphenation-aware wrapping
            wrapped_lines = wrap_text_with_hyphenation(p_line, font, column_width, dummy_draw)
            
            if wrapped_lines:
                all_lines.extend(wrapped_lines)
                
                # Mark the last line of this paragraph segment
                if p_idx == len(paragraph_lines) - 1:
                    # Only mark as paragraph end if this is the last line of the whole paragraph
                    paragraph_end_lines.add(current_line_idx + len(wrapped_lines) - 1)
                
                current_line_idx += len(wrapped_lines)
            
            # If there's another line in this same paragraph, just move to next line
            # but don't add an empty line (no paragraph break)
            if p_idx < len(paragraph_lines) - 1:
                # Here we don't add an empty line, just continue to the next line
                pass
        
        # Add paragraph break (empty line) only after complete paragraphs
        if paragraphs.index(paragraph) < len(paragraphs) - 1:
            all_lines.append("")
            current_line_idx += 1

    # Remove the last empty line if it exists
    if all_lines and all_lines[-1] == "":
        all_lines.pop()
        current_line_idx -= 1
    
    # Calculate effective top margin (accounting for header if present)
    effective_top_margin = top_margin + header_height
    
    # Calculate lines per page
    text_height = template_height - effective_top_margin - bottom_margin
    lines_per_column = math.floor(text_height / line_height)
    print(f"Lines per column: {lines_per_column}, Total lines: {len(all_lines)}")
    
    # Generate pages
    page_number = 1
    current_line = 0
    
    pbar = tqdm(total=len(all_lines), desc="Generating pages")
    
    while current_line < len(all_lines):
        # Initialize text content for this page
        page_text = []
        
        # Load a fresh template for each page
        template = Image.open(image_template_path)
        draw = ImageDraw.Draw(template)
        
        # Draw header if provided
        if header_text:
            header_y = top_margin
            
            # Calculate header x position based on alignment
            if header_alignment == 'left':
                header_x = left_margin
            elif header_alignment == 'center':
                header_bbox = header_font.getbbox(header_text)
                header_width = header_bbox[2] - header_bbox[0]
                header_x = (template_width - header_width) // 2
            elif header_alignment == 'right':
                header_bbox = header_font.getbbox(header_text)
                header_width = header_bbox[2] - header_bbox[0]
                header_x = template_width - right_margin - header_width
            else:  # Default to center
                header_bbox = header_font.getbbox(header_text)
                header_width = header_bbox[2] - header_bbox[0]
                header_x = (template_width - header_width) // 2
            
            # Draw header
            draw.text((header_x, header_y), header_text, font=header_font, fill=(0, 0, 0))
        
        # Track text positions on this page for bleedthrough
        text_positions = []
        
        if two_column:
            # Two-column layout
            left_column_start = left_margin
            right_column_start = left_margin + column_width + column_gap
            
            # Process left column
            y_position = effective_top_margin
            left_col_lines = 0
            
            # Track content for each column
            left_column_content = []
            
            # Fill left column
            while current_line < len(all_lines) and left_col_lines < lines_per_column:
                line = all_lines[current_line]
                
                # Add line to column content (even if empty)
                left_column_content.append(line)
                
                if line:  # Only draw non-empty lines
                    # Check if this is a paragraph end line
                    is_paragraph_end = current_line in paragraph_end_lines
                    
                    # Draw text based on alignment for left column
                    if alignment == 'justified':
                        x_pos, _ = draw_justified_text(draw, line, font, left_column_start, y_position, 
                                                    column_width, is_paragraph_end=is_paragraph_end)
                    else:
                        # Calculate x position based on alignment for left column
                        if alignment == 'left':
                            x_position = left_column_start
                        elif alignment == 'center':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = left_column_start + (column_width - text_width_pixels) // 2
                        elif alignment == 'right':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = left_column_start + column_width - text_width_pixels
                        
                        # Draw text
                        draw.text((x_position, y_position), line, font=font, fill=(0, 0, 0))
                        x_pos = x_position
                    
                    # Store text position for bleedthrough
                    text_positions.append((line, x_pos, y_position))
                
                y_position += line_height
                left_col_lines += 1
                current_line += 1
                pbar.update(1)
            
            # Add left column content to page text
            page_text.append({"column": "left", "content": left_column_content})
            
            # Process right column (continuation of text)
            y_position = effective_top_margin
            right_col_lines = 0
            
            # Track content for right column
            right_column_content = []
            
            # Fill right column
            while current_line < len(all_lines) and right_col_lines < lines_per_column:
                line = all_lines[current_line]
                
                # Add line to column content (even if empty)
                right_column_content.append(line)
                
                if line:  # Only draw non-empty lines
                    # Check if this is a paragraph end line
                    is_paragraph_end = current_line in paragraph_end_lines
                    
                    # Draw text based on alignment for right column
                    if alignment == 'justified':
                        x_pos, _ = draw_justified_text(draw, line, font, right_column_start, y_position, 
                                                     column_width, is_paragraph_end=is_paragraph_end)
                    else:
                        # Calculate x position based on alignment for right column
                        if alignment == 'left':
                            x_position = right_column_start
                        elif alignment == 'center':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = right_column_start + (column_width - text_width_pixels) // 2
                        elif alignment == 'right':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = right_column_start + column_width - text_width_pixels
                        
                        # Draw text
                        draw.text((x_position, y_position), line, font=font, fill=(0, 0, 0))
                        x_pos = x_position
                    
                    # Store text position for bleedthrough
                    text_positions.append((line, x_pos, y_position))
                
                y_position += line_height
                right_col_lines += 1
                current_line += 1
                pbar.update(1)
            
            # Add right column content to page text
            page_text.append({"column": "right", "content": right_column_content})
            
        else:
            # Single column layout
            y_position = effective_top_margin
            lines_added = 0
            
            # Track content for single column
            single_column_content = []
            
            while current_line < len(all_lines) and lines_added < lines_per_column:
                line = all_lines[current_line]
                
                # Add line to column content (even if empty)
                single_column_content.append(line)
                
                if line:  # Only draw non-empty lines
                    # Check if this is a paragraph end line
                    is_paragraph_end = current_line in paragraph_end_lines
                    
                    # Draw text based on alignment
                    if alignment == 'justified':
                        x_pos, _ = draw_justified_text(draw, line, font, left_margin, y_position, 
                                                     column_width, is_paragraph_end=is_paragraph_end)
                    else:
                        # Calculate x position based on alignment
                        if alignment == 'left':
                            x_position = left_margin
                        elif alignment == 'center':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = (template_width - text_width_pixels) // 2
                        elif alignment == 'right':
                            text_bbox = font.getbbox(line)
                            text_width_pixels = text_bbox[2] - text_bbox[0]
                            x_position = template_width - right_margin - text_width_pixels
                        
                        # Draw text
                        draw.text((x_position, y_position), line, font=font, fill=(0, 0, 0))
                        x_pos = x_position
                    
                    # Store text position for bleedthrough
                    text_positions.append((line, x_pos, y_position))
                
                y_position += line_height
                lines_added += 1
                current_line += 1
                pbar.update(1)
            # Add single column content to page text
            page_text.append({"column": "single", "content": single_column_content})
        
        # Store the text content for this page
        page_text_mapping[page_number] = page_text
        
        # Apply aging effects if enabled
        final_image = template
        
        # Add bleedthrough text if enabled
        if enable_bleedthrough and text_positions:
            final_image = add_bleedthrough_text(
                image=final_image, 
                text_positions=text_positions,  # Now passing position info for each line
                font=font,
                bleedthrough_offset=bleedthrough_offset,  # Use the function argument
                bleedthrough_opacity=bleedthrough_opacity,  # Use the function argument
                sepia_color=sepia_color  # Use the function argument
            )
        
        # Add ink blemishes if enabled
        if enable_ink_blemishes:
            final_image = add_ink_blemishes(
                image=final_image,
                num_blemishes=num_blemishes,  # Pass from function args
                size_range=size_range,        # Pass from function args
                color=color,                  # Pass from function args
                opacity=opacity               # Pass from function args
            )
        
        # Save the completed page
        output_path = os.path.join(output_dir, f"{page_number}.jpg")
        final_image.save(output_path)
        # print(f"Generated page {page_number}: {output_path}")
        # pbar.update(1)
        page_number += 1
    
    pbar.close()
    # Return both the page count and the page-to-text mapping
    return (page_number - 1, page_text_mapping)  # Return tuple with page count and text mapping


def modern_to_old_spanish(text):
    """
    Convert modern Spanish text to old Spanish typography and orthography (16th-18th century).
    
    This function applies the following transformations:
    1. Replace 's' with long 's' (ſ) at word beginnings and middles
    2. Replace 'z' with 'ç'
    3. Add grave accents (substitute for macrons) over vowels before 'n'
    4. Add grave accent over 'Q' before 'ue' (substitute for macron)
    5. Use historically accurate grave accents instead of modern acute accents
    6. Apply specific old spelling patterns
    """
    import re
    
    # Step 1: Process text word by word
    processed_tokens = []
    
    # Split text into words, spaces, and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]+|\s+', text)
    
    for token in tokens:
        # Skip if not a word
        if not re.match(r'^\w+$', token):
            processed_tokens.append(token)
            continue
            
        # Process the word
        word = token
        
        # 1 & 2. Transform 's' to long 's' (ſ) and 'z' to 'ç'
        new_word = ""
        for i, char in enumerate(word):
            if char.lower() == 's' and i < len(word) - 1:  # Not the last letter
                new_word += 'ſ' if char.islower() else 'S'
            elif char.lower() == 'z':
                new_word += 'ç' if char.islower() else 'Ç'
            else:
                new_word += char
        word = new_word
        
        # 3. Add grave accents over vowels before 'n' (substitute for macrons)
        new_word = ""
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i].lower() in 'aeiou' and word[i+1].lower() == 'n':
                vowel_accent_map = {
                    'a': 'à', 'e': 'è', 'i': 'ì', 'o': 'ò', 'u': 'ù',
                    'A': 'À', 'E': 'È', 'I': 'Ì', 'O': 'Ò', 'U': 'Ù'
                }
                new_word += vowel_accent_map.get(word[i], word[i])
            else:
                new_word += word[i]
            i += 1
        word = new_word
        
        # 4. Add grave accent over 'Q' before 'ue' (substitute for macron)
        if word.startswith('Q') and len(word) > 2 and word[1:3].lower() == 'ue':
            word = 'Q' + 'ù' + word[2:]
        
        processed_tokens.append(word)
    
    text = ''.join(processed_tokens)
    
    # Step 5: Fix special cases for double-s patterns
    text = text.replace('ſſ', 'ſs')
    
    # Step 6: Special orthographic replacements
    # Keep 'iò' at the end of words (already using grave accent)
    text = re.sub(r'io\b', 'iò', text)
    
    # Step 7: Common word replacements with appropriate accents
    word_replacements = {
        # Convert any modern acute accents to grave accents
        'á': 'à', 'é': 'è', 'í': 'ì', 'ó': 'ò', 'ú': 'ù',
        'Á': 'À', 'É': 'È', 'Í': 'Ì', 'Ó': 'Ò', 'Ú': 'Ù'
    }
    for modern, old in word_replacements.items():
        text = text.replace(modern, old)
    
    # Step 8: Add grave accent to preposition 'a' (already using grave accent)
    text = re.sub(r'\ba\s', 'à ', text)
    
    # Step 9: Adjust punctuation spacing
    text = re.sub(r'(?<!\s);', ' ;', text)
    
    return text


def old_to_modern_spanish(text):
    """
    Convert old Spanish typography and orthography (16th-18th century) to modern Spanish.
    
    This function ensures the final text only contains the 27 letters of the modern Spanish alphabet:
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, Ñ, O, P, Q, R, S, T, U, V, W, X, Y, Z
    
    This function applies the following transformations:
    1. Convert both 's' and long 's' (ſ) to modern 's'
    2. Convert 'ç' to 'z'
    3. Remove all accent marks (except keeping ñ)
    4. Handle Q with grave accent before 'ue'
    
    Returns:
        str: Text converted to modern Spanish orthography
    """
    import re
    
    # Step 1: Replace long s (ſ) with standard s
    text = text.replace('ſ', 's')
    
    # Step 2: Replace ç with z
    text = text.replace('ç', 'z')
    text = text.replace('Ç', 'Z')
    
    # Step 3: Process text word by word for more complex transformations
    processed_tokens = []
    
    # Split text into words, spaces, and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]+|\s+', text)
    
    for token in tokens:
        # Skip if not a word
        if not re.match(r'^\w+$', token):
            processed_tokens.append(token)
            continue
            
        # Process the word
        word = token
        
        # Step 4: Handle Q + grave accent + 'ue' pattern
        if word.startswith('Q') and len(word) > 2 and word[1] == 'ù' and word[2] == 'e':
            word = 'Que' + word[3:]
        
        processed_tokens.append(word)
    
    text = ''.join(processed_tokens)
    
    # Step 5: Remove ALL accent marks (grave and acute) except for ñ
    # Define a mapping for all accented characters to their non-accented versions
    accent_map = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
        'Á': 'A', 'À': 'A', 'Ä': 'A', 'Â': 'A',
        'É': 'E', 'È': 'E', 'Ë': 'E', 'Ê': 'E',
        'Í': 'I', 'Ì': 'I', 'Ï': 'I', 'Î': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ö': 'O', 'Ô': 'O',
        'Ú': 'U', 'Ù': 'U', 'Ü': 'U', 'Û': 'U'
    }
    
    for old, modern in accent_map.items():
        text = text.replace(old, modern)
    
    # Step 6: Fix preposition 'à' to 'a' (in case the previous step missed it)
    text = re.sub(r'\bà\s', 'a ', text)
    
    # Step 7: Normalize punctuation spacing
    text = re.sub(r'\s+;', ';', text)
    
    return text



def find_image_templates():
    """Find all image template files in the templates directory."""
    return [
        os.path.join(root, file) 
        for root, _, files in os.walk(IMAGE_TEMPLATES_DIR)
        for file in files if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


def find_font_files():
    """Find all font files in the fonts directory."""
    return [
        os.path.join(root, file) 
        for root, _, files in os.walk(FONTS_DIR) 
        for file in files if file.endswith(".ttf")
    ]


def load_text_corpus():
    """Load and concatenate all text files from the corpus directory."""
    all_texts = []
    for root, _, files in os.walk(CORPUS_DIR):
        for file in files:
            if file.lower().endswith(".txt"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    all_texts.append(f.read() + "\n\n")
    
    return "".join(all_texts)


def prepare_text_pairs(texts):
    """
    Convert texts to modern Spanish and then to old Spanish.
    
    Args:
        texts: List of text chunks
        
    Returns:
        list: List of tuples (original_text, transformed_text)
    """
    # Convert to modern Spanish
    modern_texts = list(map(old_to_modern_spanish, texts))
    
    # Transform to old Spanish
    # tuple of modern and old text
    return [(modern_text, modern_to_old_spanish(modern_text)) for modern_text in modern_texts]


def generate_random_config(image_path, font_paths):
    """
    Generate a random configuration for page rendering.
    
    Args:
        image_path: Path to the template image
        configs: Configuration dictionary
        font_paths: List of available font paths
        
    Returns:
        dict: Configuration dictionary with rendering parameters
    """
    
    # Generate random parameters
    font_path = random.choice(font_paths)
    font_size = int(random.triangular(15, 30, 20))
    margins = {
        "left": random.randint(30, 80),
        "right": random.randint(30, 80),
        "top": random.randint(40, 100),
        "bottom": random.randint(40, 100)
    }
    
    relative_image_path = os.path.relpath(image_path, ROOT)
    config = configs.get(relative_image_path, "default")
    
    # Layout settings
    alignment = random.choice(config["alignment"]) if isinstance(config["alignment"], list) else config["alignment"]
    two_column = random.choice(config["two_column"]) if isinstance(config["two_column"], list) else config["two_column"]
    column_gap = random.randint(10, 60)
    
    # Header settings
    header_text = None
    header_font_size = random.randint(40, 120)
    header_alignment = config["header_alignment"]
    header_margin_bottom = random.randint(25, 50)
    
    # Effects settings
    enable_bleedthrough = random.random() < 0.9
    enable_ink_blemishes = random.random() < 0.8
    bleedthrough_offset = random.randint(5, 20) if random.choice([True, False]) else random.randint(-20, -5)
    bleedthrough_opacity = random.uniform(0.15, 0.4)
    sepia_color = (90, 70, 50)
    num_blemishes = (10, 40)
    size_range = (1, 12)
    ink_color = (10, 5, 0)
    ink_opacity = 200
    
    return {
        "template_path": image_path,
        "font_path": font_path,
        "font_size": font_size,
        "margins": margins,
        "alignment": alignment,
        "two_column": two_column,
        "column_gap": column_gap,
        "header_text": header_text,
        "header_font_size": header_font_size,
        "header_alignment": header_alignment,
        "header_margin_bottom": header_margin_bottom,
        "effects": {
            "enable_bleedthrough": enable_bleedthrough,
            "enable_ink_blemishes": enable_ink_blemishes,
            "bleedthrough_offset": bleedthrough_offset,
            "bleedthrough_opacity": bleedthrough_opacity,
            "sepia_color": sepia_color,
            "num_blemishes": num_blemishes,
            "size_range": size_range,
            "ink_color": ink_color,
            "ink_opacity": ink_opacity
        }
    }


def format_page_text(page_content):
    """
    Format the text content from a page.
    
    Args:
        page_content: List of column data dictionaries
        
    Returns:
        str: Formatted text as a string
    """
    formatted_text = ""
    for column_data in page_content:
        column_content = column_data.get("content", [])
        non_empty_lines = [line for line in column_content if line.strip()]
        if formatted_text and non_empty_lines:
            formatted_text += "\n\n"
        formatted_text += "\n".join(non_empty_lines)
    
    return formatted_text


def process_text(args):
    """
    Process a single text chunk to generate sample pages.
    
    Args:
        args: Tuple containing (index, (original_text, transformed_text))
        
    Returns:
        list: List of sample dictionaries with metadata
    """
    idx, (modern_text, old_text) = args
    
    image_template_paths = find_image_templates()
    font_paths = find_font_files()
    
    # Randomly select image template and generate configuration
    image_template_path = random.choice(image_template_paths)
    sample_config = generate_random_config(image_template_path, font_paths)
    
    # Create temporary output directory
    temp_dir = os.path.join(DATASET_DIR, f"temp_{idx}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate old book pages
    total_pages, page_text_mapping = simulate_old_book(
        text=old_text, 
        image_template_path=sample_config["template_path"], 
        output_dir=temp_dir,
        font_path=sample_config["font_path"], 
        font_size=sample_config["font_size"], 
        left_margin=sample_config["margins"]["left"],
        right_margin=sample_config["margins"]["right"],
        top_margin=sample_config["margins"]["top"],
        bottom_margin=sample_config["margins"]["bottom"],
        alignment=sample_config["alignment"],
        two_column=sample_config["two_column"],
        column_gap=sample_config["column_gap"],
        header_text=sample_config["header_text"],
        header_font_size=sample_config["header_font_size"],
        header_alignment=sample_config["header_alignment"],
        header_margin_bottom=sample_config["header_margin_bottom"],
        enable_bleedthrough=sample_config["effects"]["enable_bleedthrough"],
        enable_ink_blemishes=sample_config["effects"]["enable_ink_blemishes"],
        bleedthrough_offset=sample_config["effects"]["bleedthrough_offset"],
        bleedthrough_opacity=sample_config["effects"]["bleedthrough_opacity"],
        sepia_color=sample_config["effects"]["sepia_color"],
        num_blemishes=sample_config["effects"]["num_blemishes"],
        size_range=sample_config["effects"]["size_range"],
        color=sample_config["effects"]["ink_color"],
        opacity=sample_config["effects"]["ink_opacity"],
    )
    
    samples = []
    
    # Process each generated page
    for page_num in range(1, total_pages + 1):
        temp_path = os.path.join(temp_dir, f"{page_num}.jpg")
        new_filename = f"text_{idx}_page_{page_num}.jpg"
        new_path = os.path.join(DATASET_DIR, new_filename)
        
        
        # Move file to main dataset directory
        os.rename(temp_path, new_path)
        
        # Format page text
        page_content = page_text_mapping.get(page_num, [])
        transformed_text_page = format_page_text(page_content)
        
        # Create sample metadata
        samples.append({
            "image_path": new_path,
            "transformed_text": transformed_text_page,
            "original_text": modern_text,
            "page_number": page_num,
            "total_pages": total_pages,
            "sample_id": f"text_{idx}_page_{page_num}",
            "config": sample_config
        })
    
    # Cleanup
    os.rmdir(temp_dir)
    
    return samples


def main():
    """Main function to create the OCR dataset."""
    # Create output directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Load and process corpus texts
    all_texts = load_text_corpus()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    text_chunks = text_splitter.split_text(all_texts)
    
    # Prepare text pairs
    texts_pairs = prepare_text_pairs(text_chunks)
    
    # Process all texts in parallel
    all_samples = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, os.cpu_count() or 1)) as executor:
        for samples in executor.map(process_text, enumerate(texts_pairs)):
            all_samples.extend(samples)
    
    # Save dataset
    dataset = {"samples": all_samples}
    dataset_json_path = os.path.join(DATASET_DIR, "dataset.json")
    with open(dataset_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    return len(all_samples)


if __name__ == "__main__":
    total_samples = main()
    print(f"Dataset generation complete. Total samples: {total_samples}")