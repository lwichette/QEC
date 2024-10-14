import os
import json

def fix_faulty_txt_file(file_path):
    """Reads a faulty txt file and converts it to a valid JSON format."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        fixed_json = []
        current_object = ""

        for line in lines:
            line = line.strip()

            # If a new object starts, process the current one (if any)
            if line.startswith("{"):
                if current_object:
                    try:
                        fixed_json.append(json.loads(current_object))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
                        return
                    current_object = ""

            # Accumulate lines into the current object
            current_object += line

        # Add the last object if exists
        if current_object:
            try:
                fixed_json.append(json.loads(current_object))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
                return

        # Write the corrected JSON to a new file with the same name but different extension
        output_file_path = file_path.replace(".txt", "_fixed.json")
        with open(output_file_path, 'w') as f:
            json.dump(fixed_json, f, indent=4)

        print(f"Fixed file written to {output_file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory(root_dir):
    """Walk through the directory structure and fix all txt files."""
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("StitchedHistogram") and file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                fix_faulty_txt_file(file_path)


root_dir = "./results/eight_vertex/bit_flip_limit/"
process_directory(root_dir)
