import argparse
import glob
import multiprocessing
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm



MAX_LENGTH_DEFAULT = 10**4

def split_text_on_full_stop(text: str, MAX_LENGTH: int) -> list:
    """
    Splits the text into chunks of at most max_length characters,
    preferring to split at full stops. If no full stop is found,
    it falls back to splitting at a newline, and if still not found,
    it splits exactly at max_length.
    """
    segments = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # If the remaining text is short enough, append and break
        if text_length - start <= MAX_LENGTH:
            segments.append(text[start:].strip())
            break
        
        # Look for the last full stop in the allowed slice
        slice_end = start + MAX_LENGTH
        segment = text[start:slice_end]
        split_index = segment.rfind('.')
        
        if split_index != -1:
            # We found a full stop, include it in the segment
            end = start + split_index + 1
        else:
            # Fallback: try to break on newline
            newline_index = segment.rfind('\n')
            if newline_index != -1:
                end = start + newline_index + 1
            else:
                # No full stop or newline; split at max_length directly
                end = slice_end
        
        # Append the found segment and update start index
        segments.append(text[start:end].strip())
        start = end

    return segments

def main():
    parser = argparse.ArgumentParser(
        description="Test segmentation of .txt files using split_text_on_full_stop"
    )
    parser.add_argument("input_file", type=str, help="Path to the .txt file to be processed")
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH_DEFAULT,
        help="Maximum length for each segment (default: %(default)s)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist.")
        return

    with open(args.input_file, "r", encoding="utf-8") as infile:
        text = infile.read()

    # Process the text using the helper function
    segments = split_text_on_full_stop(text, args.max_length)
    
    print(f"Total number of segments created: {len(segments)}")
    print("=" * 40)

    # Print each segment with some header info to visually inspect
    for idx, seg in enumerate(segments, start=1):
        print(f"Segment {idx} (length: {len(seg)} characters):")
        print(seg)
        print("-" * 40)

if __name__ == "__main__":
    main()
