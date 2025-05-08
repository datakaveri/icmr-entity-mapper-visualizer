import json
import urllib.parse
import argparse
import sys
import re

def json_to_url(json_data, base_url):
    """
    Convert JSON data to a URL with encoded parameters
    """
    try:
        # If json_data is a string, parse it, otherwise use it directly
        if isinstance(json_data, str):
            # Try to clean the JSON string in case it's malformed
            json_data = clean_json_string(json_data)
            data_dict = json.loads(json_data)
        else:
            data_dict = json_data
            
        # Convert the dictionary to a JSON string
        json_str = json.dumps(data_dict)
        
        # Create parameters dictionary with the JSON string
        params = {'data': json_str}
        
        # Combine base URL with encoded parameters
        final_url = base_url + "?" + urllib.parse.urlencode(params)
        
        return final_url
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def clean_json_string(json_str):
    """
    Try to clean and extract valid JSON from a potentially malformed string
    """
    # Try to match a complete JSON object pattern
    match = re.search(r'({[^}]*})', json_str)
    if match:
        potential_json = match.group(1)
        try:
            # Validate if this is parseable JSON
            json.loads(potential_json)
            return potential_json
        except:
            pass
    
    # If we couldn't find a valid JSON object, return the original string
    return json_str

def main():
    parser = argparse.ArgumentParser(description="Convert JSON to URL parameters")
    parser.add_argument("--json", help="JSON string or file path")
    parser.add_argument("--file", help="JSON file path")
    parser.add_argument("--base-url", default="http://localhost:8050/", 
                        help="Base URL (default: http://localhost:8050/)")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--debug", action="store_true", help="Show debugging information")
    
    args = parser.parse_args()
    
    # Get JSON data from either direct input or file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)
    elif args.json:
        json_data = args.json
        if args.debug:
            print(f"Input JSON string: {json_data}")
    else:
        # If no input specified, try to read from stdin
        try:
            json_data = json.load(sys.stdin)
        except:
            print("Error: No valid JSON input provided")
            parser.print_help()
            sys.exit(1)
    
    # Convert JSON to URL
    result = json_to_url(json_data, args.base_url)
    
    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"URL saved to {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()