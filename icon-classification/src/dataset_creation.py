import os
import json
import argparse
from utils import list_files_from_dir, extract_value_for_key

# parser = argparse.ArgumentParser(
#     description='Icon Classification Dataset Creation')

# parser.add_argument('--ui-dir', type=str, required=True,
#                     help="Path to rico-uis directory.")
# parser.add_argument('--anno-dir', type=str, required=True,
#                     help="Path to rico-semantic-annotations directory.")

# opt = parser.parse_args()

# ui_files = list_files_from_dir(
#     opt.ui_dir, extension="*.jpg", recursive=False, sort=True)
# annotations_files = list_files_from_dir(
#     opt.anno_dir, extension="*.json", recursive=False, sort=True)

# print(f"Found {len(ui_files)} unique UI files")
# print(f"Found {len(annotations_files)} unique annotations files")
