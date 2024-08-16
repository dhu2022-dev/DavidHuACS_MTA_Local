import json
import yaml

import json

# Read the input .txt file and parse the entries
def read_txt_file(file_path):
    entries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip whitespace and ignore empty lines
            stripped_line = line.strip()
            if stripped_line:
                try:
                    # Parse JSON line
                    entry = json.loads(stripped_line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")
    return entries

def write_json_file(entries, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(entries, file, indent=4, ensure_ascii=False)

# Uncomment to convert input file to jsonl
#input_file_path = 'funders_completed_checked.txt'
#output_file_path = 'funders.json'  
#entries = read_txt_file(input_file_path)

# Write the parsed entries to the output .json file
#write_json_file(entries, output_file_path)

#print(f"Successfully converted to {output_file_path}")

# Code to clean jsonl formatting issues
import json

def clean_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Remove unwanted characters (if any) and ensure correct formatting
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    # Combine lines into a single list of JSON objects
    json_objects = []
    for line in cleaned_lines:
        try:
            json_obj = json.loads(line)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid line: {line}")
    
    # Write the cleaned JSON objects to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(json_objects, outfile, indent=4)

# Usage
input_file = 'funders_doccano.jsonl'
output_file = 'cleaned_doccano.jsonl'
clean_jsonl_file(input_file, output_file)


def json_to_yaml(json_file_path, yaml_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Write the data to a YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(json_data, yaml_file, default_flow_style=False)

json_file_path = 'cleaned_doccano.jsonl' #replace this with Ram's file
yaml_file_path = 'funders.yaml'
json_to_yaml(json_file_path, yaml_file_path)
print(f"converted to yaml in {yaml_file_path}")

# Cleaning up YAML formatted file
import re

def convert_text(text):
    # Replace any HTML escape codes with their UTF-8 equivalents
    text = text.replace(r'\"', '"')  # Convert escaped quotes
    text = text.replace(r'\n', '\n')  # Convert escaped newlines
    text = re.sub(r'\\x([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), text)  # Convert hex escapes
    return text

def modify_yaml_entry(entry):
    # Convert 'label' field
    new_labels = []
    for label in entry.get('label', []):
        if len(label) == 3:
            try:
                start, end, category = label
                new_labels.append([int(start), int(end), category])
            except ValueError:
                # Skip any label that cannot be processed
                pass
    entry['label'] = new_labels
    
    # Convert 'text' field
    entry['text'] = convert_text(entry.get('text', ''))
    return entry

def process_yaml_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = yaml.safe_load(infile)

    # Modify each entry
    modified_data = [modify_yaml_entry(entry) for entry in data]

    with open(output_file, 'w') as outfile:
        yaml.safe_dump(modified_data, outfile, default_flow_style=False, allow_unicode=True)

input_file = 'funders.yaml'
output_file = 'funders_final.yaml'
process_yaml_file(input_file, output_file)