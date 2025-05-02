import os
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import json
import requests
import re
from mesop_demo_utils_v2_mod_2 import (
    extract_entities, 
    add_adarv_dict_attrs, 
    create_col_abbrs, 
    assign_categories,  # Add this import
    initialize_emb_model, 
    load_templates,
    process_snmd_cds,
    mark_condition_resources,
    remove_punctuation,
    create_valueset,
    load_fhir_data
)
from dash.exceptions import PreventUpdate
import time
import copy

# Add this utility function to your file (at the top with other utility functions)
def sanitize_for_json(text):
    """Sanitize text to be safely included in JSON strings."""
    if not isinstance(text, str):
        return str(text)
    
    # Replace all JSON-problematic characters
    return (text.replace('\\', '\\\\')
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('\t', '\\t'))

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "Entity-to-SNOMED Mapper"

# Define API endpoint
API_ENDPOINT = "http://localhost:8000/search"

# Try to load FHIR templates
try:
    templates = load_templates()
    print("FHIR templates loaded successfully")
except Exception as e:
    print(f"Error loading templates: {e}")
    templates = None

# Layout
app.layout = dbc.Container([
    html.H2("Entity-to-SNOMED Mapper", className="mt-4 mb-4"),
    
    dbc.Card([
        dbc.CardHeader("Dataset Information"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Username field
                    dbc.Label("Username:"),
                    dbc.Input(id="username", placeholder="Enter your username", type="text", className="mb-3"),
                    
                    # Dataset type dropdown
                    dbc.Label("Dataset Type:"),
                    dcc.Dropdown(
                        id="dataset-type",
                        options=[
                            {"label": "OBI", "value": "OBI"},
                            {"label": "Surveillance", "value": "Surveillance"},
                            {"label": "Research", "value": "Research"}
                        ],
                        placeholder="Select dataset type",
                        className="mb-3"
                    ),
                    
                    # File upload component
                    dbc.Label("Upload Dataset:"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status')
                ], width=12)
            ]),
        ])
    ], className="mb-4"),
    
    # Status indicators
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Processing Status", className="card-title"),
                    html.Div(id="processing-status", children="No file uploaded")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Column Stats", className="card-title"),
                    html.Div(id="column-stats")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Results section
    html.Div([
        dbc.Tabs([
            dbc.Tab(label="Column Mapping", tab_id="tab-mapping", children=[
                html.Div(id="column-mapping-table", className="mt-3")
            ]),
            dbc.Tab(label="Entity Search", tab_id="tab-entity-search", children=[
                html.Div(id="entity-search-content", className="mt-3")
            ]),
            dbc.Tab(label="Final Mapping", tab_id="tab-final-mapping", children=[
                html.Div(id="final-mapping-content", className="mt-3")
            ]),
            dbc.Tab(label="FHIR Processing", tab_id="tab-fhir-processing", children=[
                html.Div(id="fhir-processing-content", className="mt-3")
            ]),
        ], id="tabs", active_tab="tab-mapping")
    ], id="results-section", style={"display": "none"}),
    
    # Store components for data
    dcc.Store(id="stored-data"),
    dcc.Store(id="extracted-entities"),
    dcc.Store(id="adarv-entities"),
    dcc.Store(id="renamed-columns"),
    dcc.Store(id="snomed-mappings"),
    dcc.Store(id="original-filename"),  # Add this store for the original filename
    dcc.Store(id="current-category-column-index"),  # Add this store for the current category column index
    
    # Modal for entity search
    dbc.Modal([
        dbc.ModalHeader("Search SNOMED CT Codes"),
        dbc.ModalBody([
            html.Div(id="modal-entity-info"),
            html.H6("Search Results:", className="mt-3"),
            html.Div(id="modal-search-results", style={"maxHeight": "400px", "overflow": "auto"}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="modal-close", className="ml-auto"),
        ]),
    ], id="entity-search-modal", size="lg"),

    # Add this to your app.layout, after the entity-search-modal
    dbc.Modal([
        dbc.ModalHeader("Select Category"),
        dbc.ModalBody([
            html.Div(id="category-modal-column-name"),
            dcc.Dropdown(
                id="category-dropdown-modal",
                options=[
                    {'label': 'Date/Time', 'value': 'Date/Time'},
                    {'label': 'Exposure', 'value': 'Exposure'},
                    {'label': 'ID', 'value': 'ID'},
                    {'label': 'Outcome', 'value': 'Outcome'},
                    {'label': 'Social Demographic', 'value': 'Social Demographic'},
                    {'label': 'Status', 'value': 'Status'},
                    {'label': 'Symptom', 'value': 'Symptom'},
                    {'label': '', 'value': ''}  # Empty option
                ],
                clearable=False,
                style={"width": "100%"}
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Save", id="save-category-button", className="ml-auto", color="primary"),
            dbc.Button("Cancel", id="cancel-category-button", className="ml-2")
        ]),
    ], id="category-selection-modal", size="md"),

    # Add this near the end of your app.layout, before the closing brackets
    dcc.Loading(
        id="loading-fhir",
        type="circle",
        children=html.Div(id="fhir-processing-output")
    ),
], fluid=True)

# Utility function to split entities from comma-separated text
def split_entities(entity_text):
    if not entity_text or entity_text == "None":
        return []
    
    # First, clean and normalize the string
    entity_text = entity_text.strip()
    
    # Improved entity splitting
    raw_entities = re.split(r',\s*', entity_text)
    
    # Clean up and deduplicate each entity
    entities = []
    for entity in raw_entities:
        entity = entity.strip()
        if entity and entity not in entities:
            entities.append(entity)
    
    return entities

# Callback to open the category modal when a cell in the Category column is clicked
@app.callback(
    [Output("category-selection-modal", "is_open"),
     Output("category-modal-column-name", "children"),
     Output("category-dropdown-modal", "value")],
    [Input("column-mapping-datatable", "active_cell")],
    [State("column-mapping-datatable", "data"),
     State("category-selection-modal", "is_open")]
)
def toggle_category_modal(active_cell, table_data, is_open):
    if active_cell and active_cell["column_id"] == "Category":
        # Get the row
        row_idx = active_cell["row"]
        row_data = table_data[row_idx]
        
        # Get column name and current category
        col_name = row_data["Original Column"]
        current_category = row_data.get("Category", "")
        
        return True, html.H5(f"Select category for column: {col_name}"), current_category
    
    return False, "", ""

# Callback to update the current column index when the modal is opened
@app.callback(
    Output("current-category-column-index", "data"),
    [Input("category-selection-modal", "is_open"),
     Input("column-mapping-datatable", "active_cell")],
)
def store_current_column_index(is_open, active_cell):
    if is_open and active_cell and active_cell["column_id"] == "Category":
        return active_cell["row"]
    return None

# Callback to close the modal without saving
@app.callback(
    Output("category-selection-modal", "is_open", allow_duplicate=True),
    [Input("cancel-category-button", "n_clicks")],
    prevent_initial_call=True
)
def close_category_modal(n_clicks):
    if n_clicks:
        return False
    return dash.no_update

# Callback to save the selected category
@app.callback(
    [Output("category-selection-modal", "is_open", allow_duplicate=True),
     Output("column-mapping-datatable", "data", allow_duplicate=True),
     Output("snomed-mappings", "data", allow_duplicate=True)],
    [Input("save-category-button", "n_clicks")],
    [State("current-category-column-index", "data"),
     State("category-dropdown-modal", "value"),
     State("column-mapping-datatable", "data"),
     State("snomed-mappings", "data")],
    prevent_initial_call=True
)
def save_category(n_clicks, row_idx, category_value, table_data, mappings_json):
    if not n_clicks or row_idx is None:
        return False, dash.no_update, dash.no_update
    
    # Update the category in the table data
    updated_data = table_data.copy()
    col_name = updated_data[row_idx]["Original Column"]
    updated_data[row_idx]["Category"] = category_value
    
    # Update the mappings if they exist
    if mappings_json:
        try:
            mappings = json.loads(mappings_json)
            
            # Update category for this column in all entity mappings
            if col_name in mappings:
                for entity in mappings[col_name]:
                    mappings[col_name][entity]["category"] = category_value
                    
            updated_mappings = json.dumps(mappings)
        except Exception as e:
            print(f"Error updating mappings: {e}")
            updated_mappings = mappings_json
    else:
        updated_mappings = mappings_json
    
    return False, updated_data, updated_mappings

# Update the process_upload callback to also store the original filename
@app.callback(
    [Output("stored-data", "data"),
     Output("upload-status", "children"),
     Output("processing-status", "children"),
     Output("original-filename", "data")],  # Add this output
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def process_upload(contents, filename):
    if contents is None:
        return None, "", "No file uploaded", None
    
    try:
        # Decode the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Process the file based on its type
        try:
            if 'csv' in filename.lower():
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename.lower():
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return None, html.Div(['Unsupported file type']), "Error: Unsupported file type", None
        except Exception as e:
            return None, html.Div(['Error processing this file: ' + str(e)]), f"Error: {str(e)}", None
        
        # Extract the base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return df.to_json(orient='split'), html.Div(['File uploaded successfully: ', filename]), "Processing data...", base_filename
    
    except Exception as e:
        return None, html.Div(['Error: ', str(e)]), f"Error: {str(e)}", None

# Callback to process data and extract entities after upload
@app.callback(
    [Output("extracted-entities", "data"),
     Output("adarv-entities", "data"),
     Output("renamed-columns", "data"),
     Output("column-stats", "children"),
     Output("processing-status", "children", allow_duplicate=True),
     Output("results-section", "style"),
     Output("column-mapping-table", "children")],
    Input("stored-data", "data"),
    prevent_initial_call=True
)
def process_data(json_data):
    if not json_data:
        return None, None, None, "", "No data to process", {"display": "none"}, None
    
    try:
        # Load the dataframe
        df = pd.read_json(json_data, orient='split')
        columns_list = df.columns.tolist()
        
        # Step 1: Extract entities from column names using spaCy
        try:
            _, col_ents = extract_entities(df, columns_list)
            print(f"Extracted entities: {col_ents}")
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            # Fallback: create a simple entity list using column names
            col_ents = []
            for col in columns_list:
                col_ents.append({col: [col]})
        
        # Step 2: Use add_adarv_dict_attrs to get ADARV mappings
        try:
            exception_cols, adarv_mappings = add_adarv_dict_attrs(columns_list)
            print(f"ADARV mappings: {len(adarv_mappings)} columns mapped, {len(exception_cols)} exceptions")
            print(f"ADARV mapping details: {adarv_mappings}")
        except Exception as e:
            print(f"Error in ADARV dictionary attributes: {e}")
            import traceback
            traceback.print_exc()
            adarv_mappings = []
            exception_cols = columns_list
        
        # Step 3: Mark conditions and process SNOMED codes to set FHIR resource types
        try:
            # First mark condition resources
            processed_col_ents = mark_condition_resources(col_ents)
            
            # Process SNOMED codes to determine appropriate FHIR resource types
            processed_col_ents = process_snmd_cds(processed_col_ents)
            
            print(f"Processed column entities with FHIR resources: {processed_col_ents}")
        except Exception as e:
            print(f"Error processing SNOMED codes: {e}")
            import traceback
            traceback.print_exc()
            processed_col_ents = col_ents
        
        # Step 4: Create column abbreviations - SEPARATE from category assignment
        try:
            renamed_cols = create_col_abbrs(adarv_mappings, processed_col_ents)
            print(f"Renamed columns: {renamed_cols}")
        except Exception as e:
            print(f"Error creating column abbreviations: {e}")
            import traceback
            traceback.print_exc()
            
            # Manual fallback implementation for renaming
            renamed_cols = {}
            vowels = "aeiouAEIOU0123456789"
            used_names = set()
            
            # First, use the ADARV suggested names where available
            for item in adarv_mappings:
                if 'Original_Column_Name' in item and 'suggested_name' in item:
                    col_name = item['Original_Column_Name']
                    renamed_cols[col_name] = item['suggested_name']
                    used_names.add(item['suggested_name'])
            
            # For columns not in ADARV, create abbreviations following the conventions
            for col_dict in processed_col_ents:
                if not isinstance(col_dict, dict):
                    continue
                
                # Get column name
                col_keys = [key for key in col_dict.keys() if key != 'FHIR_Resource']
                if not col_keys:
                    continue
                    
                col_name = col_keys[0]
                
                # Skip if already processed from ADARV mappings
                if col_name in renamed_cols:
                    continue
                
                # Determine FHIR resource type and prefix
                fhir_resource = col_dict.get('FHIR_Resource', 'observation')
                
                # Special handling for column names that indicate specific resources
                if "condition" in col_name.lower() or "diagnosis" in col_name.lower():
                    fhir_resource = "condition"
                elif "gender" in col_name.lower():
                    fhir_resource = "Patient.Gender"
                elif "age" in col_name.lower():
                    fhir_resource = "Patient.Age"
                    
                # Get the appropriate prefix based on FHIR resource
                if fhir_resource.lower().startswith('condition'):
                    prefix = 'con'
                elif fhir_resource.lower().startswith('patient.gender'):
                    prefix = 'pat'
                elif fhir_resource.lower().startswith('patient.age'):
                    prefix = 'pat'
                elif fhir_resource.lower().startswith('patient'):
                    prefix = 'pat'
                else:
                    prefix = fhir_resource[:3].lower()
                
                # Get entities and create abbreviations
                entity_abbrs = []
                if isinstance(col_dict[col_name], list):
                    entities = col_dict[col_name]
                else:
                    entities = [col_dict[col_name]]
                
                for entity in entities:
                    if isinstance(entity, str) and len(entity) > 1:
                        # Remove vowels
                        no_vowels = "".join([char for char in entity if char.lower() not in vowels])
                        # Remove spaces and punctuation
                        no_vowels = remove_punctuation(no_vowels.replace(" ", ""))
                        # Take first 3 characters
                        entity_abbr = no_vowels[:3].lower()
                        if entity_abbr and entity_abbr not in entity_abbrs:
                            entity_abbrs.append(entity_abbr)
                
                # Create the abbreviation
                if entity_abbrs:
                    abbr_col = f"{prefix}_{entity_abbrs[0]}"
                else:
                    # Fallback: use first 3 chars of column name
                    col_abbr = "".join([char for char in col_name if char.lower() not in vowels])
                    col_abbr = col_abbr[:3].lower()
                    abbr_col = f"{prefix}_{col_abbr}"
                
                # Handle duplicates
                base_abbr = abbr_col
                counter = 1
                while abbr_col in used_names:
                    abbr_col = f"{base_abbr}{counter}"
                    counter += 1
                
                renamed_cols[col_name] = abbr_col
                used_names.add(abbr_col)
        
        # NEW STEP 5: Assign categories separately
        try:
            category_map = assign_categories(columns_list, adarv_mappings)
            print(f"Category assignments: {category_map}")
        except Exception as e:
            print(f"Error assigning categories: {e}")
            import traceback
            traceback.print_exc()
            # If category assignment fails, use empty categories
            category_map = {col: "" for col in columns_list}
        
        # Build FHIR resource map from processed entities and ADARV mappings
        fhir_resource_map = {}
        
        # First from processed entities
        for col_dict in processed_col_ents:
            if isinstance(col_dict, dict):
                col_keys = [key for key in col_dict.keys() if key != 'FHIR_Resource']
                if col_keys:
                    col_name = col_keys[0]
                    if 'FHIR_Resource' in col_dict:
                        fhir_resource_map[col_name] = col_dict['FHIR_Resource']
                        print(f"Set FHIR resource for {col_name}: {col_dict['FHIR_Resource']}")
        
        # Then from ADARV mappings (these take precedence)
        for item in adarv_mappings:
            if 'Original_Column_Name' in item and 'FHIR_Resource' in item:
                col_name = item['Original_Column_Name']
                fhir_resource = item['FHIR_Resource']
                fhir_resource_map[col_name] = fhir_resource
                print(f"Set ADARV FHIR resource for {col_name}: {fhir_resource}")
        
        # Process entities for frontend display
        formatted_entities = {}
        
        # Process entity information from GLiNER and post-processing
        for col_dict in processed_col_ents:
            try:
                if isinstance(col_dict, dict):
                    # Extract column name and FHIR resource
                    col_keys = [key for key in col_dict.keys() if key != 'FHIR_Resource']
                    if not col_keys:
                        continue
                        
                    col_name = col_keys[0]
                    
                    # Get FHIR resource type from the map or from the processed entity
                    fhir_resource = fhir_resource_map.get(col_name, col_dict.get('FHIR_Resource', 'observation'))
                    
                    # Extract entities based on format
                    if isinstance(col_dict[col_name], list):
                        entities = col_dict[col_name]
                        
                        # Handle if entities are dicts with SCT_Name
                        processed_entities = []
                        for entity in entities:
                            if isinstance(entity, dict) and 'SCT_Name' in entity:
                                processed_entities.append(entity['SCT_Name'])
                            elif isinstance(entity, str):
                                processed_entities.append(entity)
                        
                        formatted_entities[col_name] = {
                            'entities': processed_entities,
                            'fhir_resource': fhir_resource
                        }
                    else:
                        # Handle single entity
                        formatted_entities[col_name] = {
                            'entities': [str(col_dict[col_name])],
                            'fhir_resource': fhir_resource
                        }
            except Exception as e:
                print(f"Error formatting entity: {e}, {col_dict}")
                continue
        
        # Process ADARV entities - but don't include category info here
        adarv_formatted = {}
        for item in adarv_mappings:
            try:
                if not isinstance(item, dict):
                    continue
                
                col_name = item.get('Original_Column_Name', '')
                
                if not col_name:
                    continue
                    
                # Extract entities based on format
                entities = []
                
                if 'Entities' in item:
                    for entity in item['Entities']:
                        if isinstance(entity, dict) and 'Entity' in entity:
                            entities.append(entity['Entity'])
                        elif isinstance(entity, str):
                            entities.append(entity)
                
                # Get FHIR resource from item or default
                fhir_resource = item.get('FHIR_Resource', 'observation')
                
                adarv_formatted[col_name] = {
                    'entities': entities,
                    'suggested_name': item.get('suggested_name', col_name),
                    'fhir_resource': fhir_resource
                }
            except Exception as e:
                print(f"Error processing ADARV mapping: {e}")
                continue
        
        # Create mapping table with both sources
        table_data = []

        for i, col_name in enumerate(columns_list):
            try:
                # Get entities and FHIR resource from formatted entities - ONLY use NER results
                entities = []
                fhir_resource = 'observation'  # Default
                
                if col_name in formatted_entities:
                    entities = formatted_entities[col_name].get('entities', [])
                    fhir_resource = formatted_entities[col_name].get('fhir_resource', fhir_resource)
                
                # Get ADARV info if available
                adarv_info = adarv_formatted.get(col_name, {})
                adarv_entities = adarv_info.get('entities', [])
                suggested_name = adarv_info.get('suggested_name', '')
                adarv_fhir = adarv_info.get('fhir_resource', '')
                
                # ADARV FHIR resource takes precedence
                if adarv_fhir:
                    fhir_resource = adarv_fhir
                
                # Get category from the dedicated category_map
                category = category_map.get(col_name, "")
                
                # Process entities for display - use ONLY the NER entities
                processed_entities = []
                for entity in entities:  # Only use NER entities, not ADARV entities
                    if entity and entity not in processed_entities:
                        processed_entities.append(entity)
                
                # Store entities list as JSON string
                entities_json = json.dumps(processed_entities)
                
                table_data.append({
                    "Original Column": col_name,
                    "FHIR Resource": fhir_resource,
                    "Category": category,
                    "Extracted Entities": ", ".join(processed_entities) if processed_entities else "None",
                    "Actions": "Search entities",
                    "id": str(i),
                    "entities_list_json": entities_json
                })
                
            except Exception as e:
                print(f"Error creating table row for {col_name}: {e}")
                table_data.append({
                    "Original Column": col_name,
                    "FHIR Resource": "observation",
                    "Category": "",  # Empty category for error case
                    "Extracted Entities": "Error extracting entities",
                    "Actions": "Search entities",
                    "id": str(i),
                    "entities_list_json": "[]"
                })
        
        # Create data table for display - add Category column
        mapping_table = dash_table.DataTable(
            id='column-mapping-datatable',
            columns=[
                {"name": "Original Column", "id": "Original Column"},
                {"name": "FHIR Resource", "id": "FHIR Resource"},
                {"name": "Category", "id": "Category"},
                {"name": "Extracted Entities", "id": "Extracted Entities"},
                {"name": "Actions", "id": "Actions"}
            ],
            data=table_data,
            style_table={
                'overflowX': 'auto', 
                'overflowY': 'auto', 
                'maxHeight': '500px'
            },
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Actions'},
                    'cursor': 'pointer',
                    'color': 'blue',
                    'textDecoration': 'underline'
                },
                {
                    'if': {'column_id': 'Category'},
                    'cursor': 'pointer',
                    'color': 'green',
                    'textDecoration': 'underline'
                }
            ],
            tooltip_delay=0,
            tooltip_duration=None,
            cell_selectable=True,
            filter_action="native",
            sort_action="native",
        )
        
        # Stats display
        stats = html.Div([
            f"Total columns: {len(columns_list)}",
            html.Br(),
            f"Columns with entities: {len(formatted_entities)}",
            html.Br(),
            f"ADARV matched columns: {len(adarv_mappings)}"
        ])
        
        return (
            json.dumps(formatted_entities),
            json.dumps(adarv_formatted),
            json.dumps(renamed_cols),
            stats,
            "Processing complete",
            {"display": "block"},
            mapping_table
        )
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return (
            None,
            None,
            None, 
            f"Error: {str(e)}", 
            f"Error: {str(e)}", 
            {"display": "none"},
            None
        )

# Modify the open_entity_search_modal callback to check for mapped entities
@app.callback(
    [Output("entity-search-modal", "is_open"),
     Output("modal-entity-info", "children")],
    [Input("column-mapping-datatable", "active_cell")],
    [State("column-mapping-datatable", "data"),
     State("snomed-mappings", "data")]  # Add this state to access current mappings
)
def open_entity_search_modal(active_cell, table_data, mappings_json):
    if active_cell and active_cell["column_id"] == "Actions":
        # Get the row
        row_idx = active_cell["row"]
        row_data = table_data[row_idx]
        
        # Get column information
        col_name = row_data["Original Column"]
        fhir_resource = row_data["FHIR Resource"]
        category = row_data.get("Category", "")  # Get the category value
        
        # Get the entities list directly from the row data - now it's stored as JSON
        entities_list = []
        if "entities_list_json" in row_data:
            try:
                entities_list = json.loads(row_data["entities_list_json"])
            except:
                # Fallback if JSON parsing fails
                pass
        
        if not entities_list and row_data.get("Extracted Entities", "None") != "None":
            # Split entities from text with our improved split_entities function
            raw_entities = row_data["Extracted Entities"]
            entities_list = split_entities(raw_entities)
        
        if not entities_list:
            return True, html.Div([
                html.H5(f"Column: {col_name}"),
                html.H6(f"FHIR Resource: {fhir_resource}", className="text-muted"),
                # Add category information if available
                html.H6(f"Category: {category}", className="text-muted") if category else None,
                html.P("No entities were extracted for this column.")
            ])
        
        # Get already mapped entities from mappings
        mapped_entities = set()
        if mappings_json:
            try:
                mappings = json.loads(mappings_json)
                if col_name in mappings:
                    mapped_entities = set(mappings[col_name].keys())
            except:
                # If there's an error parsing mappings, continue with empty set
                pass
        
        # Create entity buttons for each individual entity - WITH SANITIZATION
        entity_buttons = []
        for i, entity in enumerate(entities_list):
            # Sanitize values for JSON
            safe_entity = sanitize_for_json(entity)
            safe_col_name = sanitize_for_json(col_name)
            safe_fhir = sanitize_for_json(fhir_resource)
            
            # Check if this entity is already mapped
            is_mapped = entity in mapped_entities
            
            entity_buttons.append(
                dbc.Button(
                    entity,
                    id={"type": "entity-button", 
                        "index": i, 
                        "col": safe_col_name,  # Use sanitized values
                        "entity": safe_entity, 
                        "fhir": safe_fhir},
                    color="success" if is_mapped else "primary",  # Use green for mapped entities
                    className="me-2 mb-2",
                    outline=is_mapped  # Add outline style to make it more visually distinct
                )
            )
        
        entity_info = [
            html.H5(f"Column: {col_name}"),
            html.H6(f"FHIR Resource: {fhir_resource}", className="text-muted"),
            # Add category information if available
            html.H6(f"Category: {category}", className="text-muted") if category else None,
            html.P("Click on an entity to search for SNOMED CT codes:"),
            html.Div(entity_buttons, style={"display": "flex", "flexWrap": "wrap", "gap": "5px"})
        ]
        
        # Filter out None values from entity_info
        entity_info = [item for item in entity_info if item is not None]
        
        return True, entity_info
    
    return False, None

# Callback to search for SNOMED codes when entity button is clicked
@app.callback(
    Output("modal-search-results", "children"),
    [Input({"type": "entity-button", "index": ALL, "col": ALL, "entity": ALL, "fhir": ALL}, "n_clicks")],
    [State({"type": "entity-button", "index": ALL, "col": ALL, "entity": ALL, "fhir": ALL}, "id")],
    prevent_initial_call=True
)
def search_snomed_codes(n_clicks, button_ids):
    # Check if any button was clicked
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks):
        return "Click an entity to search"
    
    # Get the clicked button ID with improved error handling
    try:
        button_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Add debug logging to help identify the issue
        print(f"Raw button ID string: {button_id_str}")
        
        try:
            # Try standard JSON parsing first
            button_id = json.loads(button_id_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in button ID: {e}")
            print(f"Problematic string: {button_id_str}")
            
            # Fallback: use regex to extract components
            # This is more robust for handling malformed JSON
            import re
            
            # Extract each component with regex
            type_match = re.search(r'"type":"([^"]+)"', button_id_str)
            index_match = re.search(r'"index":(\d+)', button_id_str)
            col_match = re.search(r'"col":"([^"]+)"', button_id_str)
            entity_match = re.search(r'"entity":"([^"]+)"', button_id_str)
            fhir_match = re.search(r'"fhir":"([^"]+)"', button_id_str)
            
            # Create a dictionary from the regex matches
            button_id = {
                "type": type_match.group(1) if type_match else "entity-button",
                "index": int(index_match.group(1)) if index_match else 0,
                "col": col_match.group(1) if col_match else "",
                "entity": entity_match.group(1) if entity_match else "",
                "fhir": fhir_match.group(1) if fhir_match else "observation"
            }
        
        entity = button_id["entity"]
        col_name = button_id["col"]
        fhir_resource = button_id["fhir"]
        
    except Exception as e:
        return html.Div([
            html.P(f"Error parsing button data: {str(e)}"),
            html.P(f"Please try again or choose a different entity.")
        ])
    
    try:
        # Make API request to search for SNOMED codes
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 10
        }
        
        # Make request to the API
        response = requests.post(API_ENDPOINT, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Safely parse JSON response
                results = json.loads(response.text)
                
                # For debugging - log the first part of the response
                print(f"API response first 100 chars: {response.text[:100]}")
                
                # Validation check - make sure results is a list
                if not isinstance(results, list):
                    return html.P(f"Invalid response format. Expected a list of results for '{entity}'.")
                
                if not results:
                    return html.P(f"No SNOMED CT codes found for '{entity}'")
                
                # Create a table of results with select buttons
                result_rows = []
                
                for i, result in enumerate(results):
                    # Validate the result has all required fields
                    if not all(k in result for k in ["conceptid", "conceptid_name", "toplevelhierarchy_name"]):
                        continue
                    
                    # Format the row
                    row = [
                        html.Tr([
                            html.Td(result["conceptid"]),
                            html.Td(result["conceptid_name"]),
                            html.Td(result["toplevelhierarchy_name"]),
                            html.Td(f"{result.get('rrf_score', 0):.3f}"),
                            html.Td(
                                dbc.Button(
                                    "Select",
                                    id={
                                        "type": "select-match",
                                        "index": i,
                                        "entity": entity,
                                        "col": col_name,
                                        "concept_id": result["conceptid"],
                                        "concept_name": result["conceptid_name"],
                                        "fhir_resource": fhir_resource or result.get("fhir_resource", "observation")
                                    },
                                    color="success",
                                    size="sm"
                                )
                            )
                        ])
                    ]
                    result_rows.extend(row)
                
                if not result_rows:
                    return html.P(f"No valid results found for '{entity}'")
                
                return html.Div([
                    html.P(f"Search results for '{entity}':"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("SNOMED ID"),
                                    html.Th("SNOMED Name"),
                                    html.Th("Hierarchy"),
                                    html.Th("Score"),
                                    html.Th("Action")
                                ])
                            ),
                            html.Tbody(result_rows)
                        ],
                        className="table table-striped table-bordered"
                    )
                ])
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                print(f"JSON parsing error: {e}")
                print(f"Response content: {response.text[:200]}...")  # First 200 chars for debugging
                return html.Div([
                    html.P(f"Error parsing search results for '{entity}'"),
                    html.P("The API returned an invalid response. Please try again later.")
                ])
            
        else:
            # Handle API error
            error_message = f"API error ({response.status_code})"
            try:
                error_data = response.json()
                if isinstance(error_data, dict) and "detail" in error_data:
                    error_message += f": {error_data['detail']}"
            except:
                # If we can't parse the response as JSON, include part of the response text
                error_message += f": {response.text[:100]}..."
                
            return html.P(f"Error: {error_message}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.P(f"Error searching for SNOMED codes: {str(e)}"),
            html.P("Please try again or contact support if the issue persists.")
        ])

# Callback to handle selecting a match
@app.callback(
    [Output("snomed-mappings", "data", allow_duplicate=True),
     Output("final-mapping-content", "children", allow_duplicate=True),
     Output("tabs", "active_tab")],
    [Input({"type": "select-match", "index": ALL, "entity": ALL, "col": ALL, 
          "concept_id": ALL, "concept_name": ALL, "fhir_resource": ALL}, "n_clicks")],
    [State({"type": "select-match", "index": ALL, "entity": ALL, "col": ALL, 
          "concept_id": ALL, "concept_name": ALL, "fhir_resource": ALL}, "id"),
     State("snomed-mappings", "data"),
     State("column-mapping-datatable", "data"),
     State("renamed-columns", "data"),
     State("modal-search-results", "children")],  # Add this state to access search results
    prevent_initial_call=True
)

def handle_match_selection(n_clicks, button_ids, mappings_json, table_data, renamed_cols_json, search_results):
    # Check if any button was clicked
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks):
        return mappings_json, dash.no_update, dash.no_update
    
    try:
        # Find which button was clicked
        clicked_idx = None
        for i, clicks in enumerate(n_clicks):
            if clicks and clicks > 0:
                clicked_idx = i
                break
        
        if clicked_idx is None:
            return mappings_json, dash.no_update, dash.no_update
        
        # Get the information directly from the button_ids list
        button_data = button_ids[clicked_idx]
        
        # Extract data from the button
        entity = button_data["entity"]
        col_name = button_data["col"]
        concept_id = button_data["concept_id"]
        concept_name = button_data["concept_name"]
        fhir_resource = button_data.get("fhir_resource", "observation")
        
        # Extract the best matching term from comma-separated concept_name using cosine similarity
        best_term = concept_name
        if ',' in concept_name:
            try:
                # Import sentence transformer for cosine similarity
                from sentence_transformers import SentenceTransformer
                import numpy as np
                
                # Load the model - using a medical-specific model if available
                model = SentenceTransformer('paraphrase-mpnet-base-v2')  # You can replace with a medical domain model
                
                # Split the concept name by commas
                terms = [term.strip() for term in concept_name.split(',')]
                
                # Encode the entity and all terms
                entity_embedding = model.encode([entity], show_progress_bar=False)[0]
                term_embeddings = model.encode(terms, show_progress_bar=False)
                
                # Calculate cosine similarity between entity and each term
                similarities = []
                for i, term_embedding in enumerate(term_embeddings):
                    # Cosine similarity calculation
                    similarity = np.dot(entity_embedding, term_embedding) / (
                        np.linalg.norm(entity_embedding) * np.linalg.norm(term_embedding)
                    )
                    similarities.append(similarity)
                
                # Find term with highest similarity
                best_idx = np.argmax(similarities)
                best_term = terms[best_idx]
                print(f"Entity: {entity}, Best match: {best_term}, Similarity: {similarities[best_idx]}")
                
                # For debugging
                for i, term in enumerate(terms):
                    print(f"Term: {term}, Similarity: {similarities[i]}")
            
            except Exception as e:
                print(f"Error in cosine similarity calculation: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to first term if cosine similarity fails
                best_term = terms[0]
        
        # Get category from table_data
        category = ""
        for row in table_data:
            if row["Original Column"] == col_name:
                category = row.get("Category", "")
                break

        # Update mappings to store SNOMED information
        mappings = {}
        if mappings_json:
            try:
                mappings = json.loads(mappings_json)
            except json.JSONDecodeError:
                print("Error parsing mappings JSON")
        
        # Create column mapping if it doesn't exist
        if col_name not in mappings:
            mappings[col_name] = {}
        
        mappings[col_name][entity] = {
            "snomed_id": concept_id,
            "snomed_name": best_term,
            "snomed_full_name": concept_name,
            "fhir_resource": fhir_resource,
            "category": category
        }

        # Generate the abbreviated column names dynamically
        column_renamed_mapping = {}
        for col, entities_map in mappings.items():
            # Extract FHIR resource from the first entity (they should all be the same for a column)
            first_entity = next(iter(entities_map.values())) if entities_map else {}
            fhir_resource = first_entity.get("fhir_resource", "observation")
            fhir_prefix = fhir_resource[:3].lower()
            
            # Collect SNOMED names for each entity
            snomed_prefixes = []
            for entity, mapping in entities_map.items():
                snomed_name = mapping.get("snomed_name", "").strip()
                if snomed_name:
                    snomed_prefixes.append(snomed_name[:3].lower())
            
            # Generate the abbreviated column name
            if snomed_prefixes:
                abbr_col = f"{fhir_prefix}_{'_'.join(snomed_prefixes)}"
            else:
                # Fallback if no SNOMED names available
                abbr_col = f"{fhir_prefix}_{col[:3].lower()}"
            
            column_renamed_mapping[col] = abbr_col

        # Store the abbreviated name in each entity's mapping for reference
        for col, entities_map in mappings.items():
            abbr_col = column_renamed_mapping.get(col, col)
            for entity in entities_map:
                mappings[col][entity]["renamed_column"] = abbr_col

        # Create the final mapping table with all entities - use the dynamically generated abbreviated names
        table_data = []
        for col, entities_map in mappings.items():
            # Get the abbreviated column name
            first_entity = next(iter(entities_map.values())) if entities_map else {}
            abbr_col = first_entity.get("renamed_column", col)
            
            # Add a row for each mapping of this column
            for entity, mapping in entities_map.items():
                table_data.append({
                    "Column": col,
                    "Abbreviated Column": abbr_col,  # Use the abbreviated name here
                    "Entity": entity,
                    "SNOMED ID": mapping["snomed_id"],
                    "SNOMED Name": mapping["snomed_name"],
                    "SNOMED Full Name": mapping.get("snomed_full_name", mapping["snomed_name"]),
                    "FHIR Resource": mapping["fhir_resource"],
                    "Category": mapping.get("category", "")
                })

        # Update the final table columns to show "Abbreviated Column" instead of "Renamed Column"
        final_table = dash_table.DataTable(
            id='final-mapping-datatable',
            columns=[
                {"name": "Column", "id": "Column"},
                {"name": "Abbreviated Column", "id": "Abbreviated Column"},  # Updated name
                {"name": "Entity", "id": "Entity"},
                {"name": "SNOMED ID", "id": "SNOMED ID"},
                {"name": "SNOMED Name", "id": "SNOMED Name"},
                {"name": "SNOMED Full Name", "id": "SNOMED Full Name"},
                {"name": "FHIR Resource", "id": "FHIR Resource"},
                {"name": "Category", "id": "Category"}
            ],
            data=table_data,
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '500px'
            },
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            filter_action="native",
            sort_action="native",
            export_format="csv",
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                    if column == "SNOMED Full Name"
                } for row in table_data
            ],
            tooltip_duration=None,
        )
        
        # Create a summary view of the mappings
        column_summary = []
        for col, entities_map in mappings.items():
            entity_items = []
            
            # Get category and renamed column for this column
            first_entity = next(iter(entities_map.values())) if entities_map else {}
            col_category = first_entity.get("category", "")
            renamed_col = first_entity.get("renamed_column", col)
            
            for entity, mapping in entities_map.items():
                entity_items.append(html.Div([
                    html.Span(f"{entity}: ", className="font-weight-bold"),
                    html.Span(f"{mapping['snomed_id']} - {mapping['snomed_name']}"),
                    html.Span(f" ({mapping['fhir_resource']})", className="text-muted")
                ], className="mb-1"))
            
            column_summary.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span(f"Original: {col}"),
                            html.Br(),
                            html.Span(f"Renamed: {renamed_col}", className="text-info")
                        ]),
                        html.Span(
                            dbc.Badge(col_category, color="info", className="ml-2 float-right"),
                            style={"marginTop": "5px"}
                        ) if col_category else None
                    ]),
                    dbc.CardBody(entity_items)
                ], className="mb-3")
            )
        
        # Add the summary view above the data table
        export_section = html.Div([
            html.H5("SNOMED CT Mappings Summary"),
            html.Div(column_summary) if column_summary else html.P("No mappings created yet."),
            
            html.H5("All Mappings", className="mt-4"),
            final_table,
            
            html.Div([
                dbc.Button("Export Mappings", id="export-button", color="primary", className="mt-3"),
                dcc.Download(id="download-mappings")
            ])
        ])
        
        return json.dumps(mappings), export_section, "tab-final-mapping"
    except Exception as e:
        print(f"Error in match selection: {e}")
        import traceback
        traceback.print_exc()
        return mappings_json, dash.no_update, dash.no_update


# Callback to close the modal
@app.callback(
    Output("entity-search-modal", "is_open", allow_duplicate=True),
    Input("modal-close", "n_clicks"),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    return False

# Update the export mappings callback to use username and dataset type in the filename
@app.callback(
    Output("download-mappings", "data"),
    Input("export-button", "n_clicks"),
    [State("snomed-mappings", "data"),
     State("username", "value"),
     State("dataset-type", "value"),
     State("original-filename", "data")],
    prevent_initial_call=True
)
def export_mappings(n_clicks, mappings_json, username, dataset_type, original_filename):
    if not mappings_json:
        return None
    
    try:
        mappings = json.loads(mappings_json)
        
        # Convert to DataFrame for export
        rows = []
        for col, entities_map in mappings.items():
            for entity, mapping in entities_map.items():
                rows.append({
                    "Column": col,
                    "Renamed_Column": mapping.get("renamed_column", col),
                    "Entity": entity,
                    "SNOMED_ID": mapping["snomed_id"],
                    "SNOMED_Name": mapping["snomed_name"],
                    "SNOMED_Full_Name": mapping.get("snomed_full_name", mapping["snomed_name"]),
                    "FHIR_Resource": mapping["fhir_resource"],
                    "Category": mapping.get("category", "")
                })
        
        # Create the filename with username and dataset type
        username = username or "user"
        dataset_type = dataset_type or "data"
        original_filename = original_filename or "file"
        
        # Clean the filename components
        username = re.sub(r'[^a-zA-Z0-9]', '_', username)
        dataset_type = re.sub(r'[^a-zA-Z0-9]', '_', dataset_type)
        original_filename = re.sub(r'[^a-zA-Z0-9]', '_', original_filename)
        
        # Generate the custom filename
        custom_filename = f"{username}_{dataset_type}_{original_filename}_snomed_mappings.csv"
        
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, custom_filename, index=False)
    except Exception as e:
        print(f"Error exporting mappings: {e}")
        return None

# Initialize values for store components to prevent errors
@app.callback(
    Output("snomed-mappings", "data"),
    Input("stored-data", "data")
)
def initialize_mappings(data):
    return json.dumps({})

# Init final mapping content
@app.callback(
    Output("final-mapping-content", "children"),
    Input("stored-data", "data")
)
def initialize_final_mapping(data):
    return html.P("Select matches from the Entity Search tab to build your final mapping.")

# Content for entity search tab
@app.callback(
    Output("entity-search-content", "children"),
    Input("extracted-entities", "data")
)
def update_entity_search_tab(entities_json):
    if not entities_json:
        return html.P("No entities extracted yet.")
    
    return html.Div([
        html.P("Click on a column in the Column Mapping tab to search for entities."),
        html.Hr(),
        html.P([
            html.Strong("Instructions:"), 
            html.Br(),
            "1. In the Column Mapping tab, click on the 'Actions' column for any row",
            html.Br(),
            "2. Select an entity from the popup to search for SNOMED codes",
            html.Br(),
            "3. Choose the best match from the search results",
            html.Br(),
            "4. View and export your final mappings from the Final Mapping tab"
        ])
    ])

# Add this callback to populate the FHIR processing tab content
@app.callback(
    Output("fhir-processing-content", "children"),
    [Input("stored-data", "data"),
     Input("snomed-mappings", "data")]
)
def update_fhir_processing_tab(json_data, mappings_json):
    if not json_data or not mappings_json:
        return html.Div([
            html.P("Upload a dataset and create SNOMED mappings first."),
            html.P("Once you've mapped your entities to SNOMED codes in the previous tabs, you can generate FHIR resources here.")
        ])
    
    try:
        mappings = json.loads(mappings_json)
        if not mappings:
            return html.Div([
                html.P("No mappings available. Please create mappings in the Entity Search tab first."),
            ])
            
        return html.Div([
            html.H5("Generate FHIR Resources"),
            html.P("Create FHIR resources from your dataset using the SNOMED mappings you've defined."),
            html.Hr(),
            
            dbc.Form([
                # Replace FormGroup with simple div + label + input pattern
                html.Div([
                    dbc.Label("Dataset Name (for Group resource):", html_for="dataset-name-input"),
                    dbc.Input(id="dataset-name-input", type="text", placeholder="Enter a name for this dataset"),
                ], className="mb-3"),
                
                html.Div([
                    dbc.Label("FHIR Server URL:", html_for="fhir-server-url"),
                    dbc.Input(id="fhir-server-url", type="text", value="http://65.0.127.208:30007/fhir", placeholder="Enter FHIR server URL"),
                ], className="mb-3"),
                
                html.Div([
                    dbc.Checkbox(id="enable-fhir-upload", className="me-2"),
                    dbc.Label("Enable FHIR server upload (otherwise just generate bundle)", html_for="enable-fhir-upload"),
                ], className="mb-3"),
                
                dbc.Button("Generate FHIR Resources", id="generate-fhir-button", color="success", className="mt-3"),
            ]),
            
            html.Div(id="fhir-process-status", className="mt-3"),
            html.Div(id="fhir-download-section", className="mt-3"),
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Error preparing FHIR processing tab: {str(e)}"),
        ])

# Callback to process the data and generate FHIR resources
@app.callback(
    [Output("fhir-process-status", "children"),
     Output("fhir-download-section", "children"),
     Output("fhir-processing-output", "children")],
    [Input("generate-fhir-button", "n_clicks")],
    [State("stored-data", "data"),
     State("snomed-mappings", "data"),
     State("dataset-name-input", "value"),
     State("enable-fhir-upload", "checked"),
     State("fhir-server-url", "value")]
)
def generate_fhir_resources(n_clicks, json_data, mappings_json, dataset_name, enable_upload, server_url):
    # Prevent the callback from running on page load
    if not n_clicks:
        raise PreventUpdate
    
    if not json_data or not mappings_json:
        return html.P("No data or mappings available."), None, None
    
    if not dataset_name:
        dataset_name = f"dataset_{int(time.time())}"
    
    try:
        # Load the original dataframe
        df = pd.read_json(json_data, orient='split')
        mappings = json.loads(mappings_json)
        
        # Initialize attributes for all columns
        for col_name in df.columns:
            # Create attrs dictionary if it doesn't exist
            if not hasattr(df[col_name], 'attrs'):
                df[col_name].attrs = {}
            
            # Initialize with default values to prevent KeyErrors
            df[col_name].attrs.setdefault('FHIR_Resource', 'observation')
            df[col_name].attrs.setdefault('Entities', [])
            df[col_name].attrs.setdefault('valueSet', 'valueString')
            df[col_name].attrs.setdefault('valueModifier', None)
        
        # Apply mappings to the dataframe by adding attributes
        for col_name, entities_map in mappings.items():
            if col_name not in df.columns:
                continue
                
            # Get the first entity mapping to determine FHIR resource type
            try:
                first_entity_key = next(iter(entities_map))
                first_entity = entities_map[first_entity_key]
                fhir_resource = first_entity.get("fhir_resource", "observation")
                
                # Set FHIR resource type as attribute
                df[col_name].attrs["FHIR_Resource"] = fhir_resource
                
                # Collect entity information for the column
                entities_list = []
                for entity_name, entity_data in entities_map.items():
                    # Create properly structured entity dictionary
                    entity_dict = {
                        'Entity': entity_name,
                        'SCT_Name': entity_data.get('snomed_name', ''),
                        'SCT_ID': entity_data.get('snomed_id', '')
                    }
                    entities_list.append(entity_dict)
                
                # Set entities attribute
                df[col_name].attrs["Entities"] = entities_list
            except Exception as e:
                print(f"Error processing mappings for column {col_name}: {e}")
                # If there's an error, keep the default attributes already set
        
        # Create valueSet attributes based on data types
        df = create_valueset(df)
        
        # For debugging: verify that all columns have the necessary attributes
        for col_name in df.columns:
            print(f"Column {col_name} attributes:")
            for attr_name, attr_value in df[col_name].attrs.items():
                print(f"  {attr_name}: {attr_value}")
        
        # Generate FHIR bundle without uploading to server
        config = {
            "upload_to_server": enable_upload,
            "server_url": server_url if enable_upload else None
        }
        
        # Use the load_fhir_data function to generate FHIR resources
        # from mesop_demo_utils_v2_mod import load_fhir_data
        fullbundle, sample_bundle = load_fhir_data(df, dataset_name)
        
        # Count resources by type
        resource_counts = {}
        for bundle in fullbundle:
            for entry in bundle["entry"]:
                resource_type = entry["resource"]["resourceType"]
                if resource_type in resource_counts:
                    resource_counts[resource_type] += 1
                else:
                    resource_counts[resource_type] = 1
        
        # Create a status message
        status_message = [
            html.H5("FHIR Resources Generated Successfully"),
            html.P(f"Total bundles: {len(fullbundle)}"),
            html.Ul([
                html.Li(f"{resource_type}: {count} resources") 
                for resource_type, count in resource_counts.items()
            ])
        ]
        
        if enable_upload:
            status_message.append(html.P(f"Resources were uploaded to {server_url}", className="text-success"))
        else:
            status_message.append(html.P("Resources were generated but not uploaded to the FHIR server", className="text-warning"))
        
        # Create download buttons
        download_section = html.Div([
            dbc.Button("Download Full Bundle", id="download-full-bundle", color="primary", className="me-2"),
            dbc.Button("Download Sample Bundle", id="download-sample-bundle", color="info"),
            dcc.Download(id="download-bundle-data")
        ])
        
        # Store bundles in a hidden div for download
        hidden_storage = html.Div([
            html.Div(id="full-bundle-storage", children=json.dumps(fullbundle), style={"display": "none"}),
            html.Div(id="sample-bundle-storage", children=json.dumps(sample_bundle), style={"display": "none"})
        ])
        
        return status_message, download_section, hidden_storage
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return [
            html.H5("Error Generating FHIR Resources", className="text-danger"),
            html.P(str(e)),
            html.Details([
                html.Summary("Error Details"),
                html.Pre(error_details)
            ], className="text-danger")
        ], None, None

# Callbacks for downloading the bundles
@app.callback(
    Output("download-bundle-data", "data"),
    [Input("download-full-bundle", "n_clicks"),
     Input("download-sample-bundle", "n_clicks")],
    [State("full-bundle-storage", "children"),
     State("sample-bundle-storage", "children"),
     State("dataset-name-input", "value")]
)
def download_bundle(full_clicks, sample_clicks, full_bundle_json, sample_bundle_json, dataset_name):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0] 
    
    if not dataset_name:
        dataset_name = f"dataset_{int(time.time())}"
    
    if button_id == "download-full-bundle":
        return dict(
            content=full_bundle_json,
            filename=f"{dataset_name}_full_bundle.json",
            type="application/json"
        )
    elif button_id == "download-sample-bundle":
        return dict(
            content=sample_bundle_json,
            filename=f"{dataset_name}_sample_bundle.json",
            type="application/json"
        )
    
    raise PreventUpdate

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)