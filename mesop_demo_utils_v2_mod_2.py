import re
import csv
import copy
import json
import time
import math
import json
import uuid
import click
import spacy
import string
import urllib
import psycopg2
import requests
import datetime
import traceback
import numpy as np
import mesop as me
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from transformers import pipeline
from psycopg2.extras import DictCursor
from dataclasses import dataclass, field
import multiprocessing
from functools import partial
from math import ceil
from sentence_transformers import SentenceTransformer
from hierarchy_transformers import HierarchyTransformer


def load_templates():
    ## Reading FHIR Resource JSON Templates
    with open("./templates/Patient.json", "r") as f:
        patient_template = json.load(f)
    with open("./templates/Age.json", "r") as f:
        age_template = json.load(f)
    with open("./templates/Observation.json", "r") as f:
        obs_template = json.load(f)
    with open("./templates/Location.json", "r") as f:
        loc_template = json.load(f)
    with open("./templates/Condition.json", "r") as f:
        cond_template = json.load(f)
    with open("./templates/NutritionIntake.json", "r") as f:
        ni_template = json.load(f)
    with open("./templates/Encounter.json", "r") as f:
        enc_template = json.load(f)
    with open("./templates/Group.json", "r") as f:
        grp_template = json.load(f)
    return patient_template, age_template, obs_template, loc_template, cond_template, ni_template, enc_template, grp_template

def extract_entities(data, columns_list):
    """Extract entities from column names using spaCy's en_sci_core_md model."""
    import spacy
    
    # Initialize spaCy model
    try:
        nlp = spacy.load("en_core_sci_md")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        # Fallback if model loading fails - just use column names as entities
        col_ents = []
        for col in columns_list:
            col_ents.append({col: [col]})  # Use column name as the entity
        return data, col_ents
    
    col_ents = []
    for col in columns_list:
        try:
            # Process text using spaCy
            doc = nlp(col)
            
            # Extract entities from spaCy's processing
            ents = []
            for ent in doc.ents:
                ents.append(str(ent.text))
            
            # If no entities found, use the column name itself
            if not ents:
                ents = [col]
                
            col_ents.append({col: ents})
            
        except Exception as e:
            print(f"Error extracting entities for column {col}: {e}")
            # On error, use column name as fallback
            col_ents.append({col: [col]})
    
    # We don't remove columns without entities since we now ensure every column has at least one entity
    return data, col_ents

def mark_condition_resources(list_of_dicts):
    for outer_dict in list_of_dicts:
        for key, value in outer_dict.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                if any(inner_dict.get('FHIR_Resource') == 'condition' for inner_dict in value):
                    for inner_dict in value:
                        inner_dict['FHIR_Resource'] = 'condition'
    
    return list_of_dicts

def process_snmd_cds(list_of_dicts):
    for outer_dict in list_of_dicts:
        updates = {}
        for key, value in outer_dict.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                if any(inner_dict.get('FHIR_Resource') == 'condition' for inner_dict in value):
                    updates['FHIR_Resource'] = 'condition'
                elif any(inner_dict.get('FHIR_Resource') == 'Patient.Age' for inner_dict in value):
                    updates['FHIR_Resource'] = 'Patient.Age'
                elif any(inner_dict.get('FHIR_Resource') == 'Patient.Gender' for inner_dict in value):
                    updates['FHIR_Resource'] = 'Patient.Gender'
                else:
                    updates['FHIR_Resource'] = 'observation'
        
        outer_dict.update(updates)
    return list_of_dicts

def add_adarv_dict_attrs(col_lst):
    cur, encoder = initialize_emb_model()
    adarv_snmd_cds = []
    exception_col_list = []
    uncertainty_matches = []  # For matches between confidence thresholds
    
    for col in tqdm(col_lst):
        col_cds = []
        vector = encoder.encode(col)
        vector = '[' + ', '.join(map(str, vector)) + ']'
        
        # Updated query to use adarv_data_dict_768_2 and include the category field
        query = f"SELECT *, (1 - (embeddings <=> '{vector}')) as score FROM adarv_data_dict_768_2 order by score desc limit 3;"
        cur.execute(query)
        matches = cur.fetchall()
        
        # Check if we have a high-confidence match
        best_match = dict(matches[0]) if matches else None
        if (best_match and best_match['score'] > 0.3):  # Using threshold of 0.9
            # Process high-confidence match
            for sct in best_match['snomed_codes'].split(', '):
                SCT_Name = re.findall(r'[A-Za-z ]+', sct)
                SCT_ID = re.findall(r'\d+', sct)
                if SCT_Name and SCT_Name[0].strip().lower() not in ['observation', 'finding']:
                    if len(SCT_Name) > 0 and len(SCT_ID) > 0:
                        col_cds.append({'Entity': sct,
                                       'SCT_Name': " ".join(SCT_Name).strip(),
                                       'SCT_ID': SCT_ID})
                                       
            if len(col_cds) > 0:
                if best_match['fhir_resource'] == "patient.gender":
                    fhir_resource = "Patient.Gender"
                else:
                    fhir_resource = best_match['fhir_resource']
                adarv_snmd_cds.append({
                    'Original_Column_Name': col, 
                    'Entities': col_cds,
                    'FHIR_Resource': fhir_resource, 
                    'suggested_name': best_match['suggested_name'],
                    'category': best_match['category']  # Include the new category field
                })
            else:
                exception_col_list.append(col)
        else:
            # No good match found
            exception_col_list.append(col)
            
    return exception_col_list, adarv_snmd_cds

def assign_categories(columns_list, adarv_mappings, similarity_threshold=0.3):
    """
    Assign categories to columns based on ADARV mappings and fallback to similarity search
    with a lower threshold for categorization vs. column renaming.
    
    Args:
        columns_list: List of column names
        adarv_mappings: ADARV mappings containing category info (from high-confidence matches)
        similarity_threshold: Lower similarity threshold specifically for category assignment
        
    Returns:
        Dictionary mapping column names to categories
    """
    category_map = {}
    
    # First, use categories from existing ADARV mappings (high confidence matches)
    for item in adarv_mappings:
        if isinstance(item, dict) and 'Original_Column_Name' in item:
            col_name = item['Original_Column_Name']
            # Only use category from ADARV data if available - otherwise leave blank
            if 'category' in item and item['category']:
                category_map[col_name] = item['category']
            else:
                category_map[col_name] = ""
    
    # For remaining columns not in ADARV mappings, use similarity search with lower threshold
    remaining_columns = [col for col in columns_list if col not in category_map]
    
    if remaining_columns:
        # Initialize database connection and encoder model
        cur, encoder = initialize_emb_model()
        
        for col in tqdm(remaining_columns, desc="Assigning categories"):
            # Generate embedding for column name
            try:
                vector = encoder.encode(col)
                vector = '[' + ', '.join(map(str, vector)) + ']'
                
                # Query with lower threshold specifically for category assignment
                query = f"SELECT *, (1 - (embeddings <=> '{vector}')) as score FROM adarv_data_dict_768_2 order by score desc limit 1;"
                cur.execute(query)
                match = cur.fetchone()
                
                # Use match if it meets the lower categorization threshold
                if match and match['score'] > similarity_threshold:
                    if match['category']:
                        category_map[col] = match['category']
                    else:
                        category_map[col] = ""
                else:
                    # No matching category found even with lower threshold
                    category_map[col] = ""
            except Exception as e:
                print(f"Error finding category for column '{col}': {str(e)}")
                category_map[col] = ""
    
    return category_map

def initialize_emb_model():
    conn_params = {
        'dbname': 'postgresdb',
        'user': 'postgres',
        'password': 'timescaledbpg',
        'host': '65.0.127.208',  
        'port': '32588' 
    }

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=DictCursor)

    encoder =  SentenceTransformer("paraphrase-mpnet-base-v2") #paraphrase-mpnet-base-v2, HierarchyTransformer.from_pretrained("Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT"), SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb")
    return cur, encoder

## WORKING EXACT + KEYWORD + SEMATIC + FUZZY (ALL INCL. IN RRF)

def add_pgvector_attrs(col_ents):
    """Enhanced version with hybrid search using weighted RRF for better entity mapping.
    Returns both SNOMED codes and top matches for each column entity."""
    cur, encoder = initialize_emb_model()
    snmd_cds = []
    
    # Create a dictionary to store top matches by column name
    column_top_matches = {}
    
    for col in tqdm(col_ents):
        col_name = list(col.keys())[0]
        col_cds = []
        st = set(list(col.values())[0])
        
        # Initialize top matches for this column
        column_top_matches[col_name] = []
        
        for ent in st:
            # Generate embedding for entity
            vector = encoder.encode(ent)
            vector_str = '[' + ', '.join(map(str, vector)) + ']'
            
            # Updated query with weighted RRF combination approach
            query = """
            WITH exact_match AS (
                SELECT 
                    conceptid,
                    conceptid_name,
                    toplevelhierarchy_name,
                    fhir_resource,
                    1.0 AS original_score,
                    1 AS rank,
                    'exact' AS match_type
                FROM snomed_ct_codes_768
                WHERE LOWER(conceptid_name) = LOWER(%s)
            ),
            fuzzy_match AS (
                SELECT 
                    conceptid,
                    conceptid_name,
                    toplevelhierarchy_name,
                    fhir_resource,
                    word_similarity(conceptid_name, %s) AS original_score,
                    ROW_NUMBER() OVER(ORDER BY word_similarity(conceptid_name, %s) DESC) AS rank,
                    'fuzzy' AS match_type
                FROM snomed_ct_codes_768
                WHERE word_similarity(conceptid_name, %s) > 0.4
                LIMIT 30
            ),
            keyword_match AS (
                SELECT 
                    conceptid,
                    conceptid_name,
                    toplevelhierarchy_name,
                    fhir_resource,
                    ts_rank(to_tsvector('english', conceptid_name), to_tsquery('english', %s)) AS original_score,
                    ROW_NUMBER() OVER(ORDER BY ts_rank(to_tsvector('english', conceptid_name), to_tsquery('english', %s)) DESC) AS rank,
                    'keyword' AS match_type
                FROM snomed_ct_codes_768
                WHERE to_tsvector('english', conceptid_name) @@ to_tsquery('english', %s)
                LIMIT 30
            ),
            vector_match AS (
                SELECT 
                    conceptid,
                    conceptid_name,
                    toplevelhierarchy_name,
                    fhir_resource,
                    1.0 / (1 + (embeddings <=> %s::vector)) AS original_score,
                    ROW_NUMBER() OVER(ORDER BY embeddings <=> %s::vector) AS rank,
                    'vector' AS match_type
                FROM snomed_ct_codes_768
                LIMIT 30
            ),
            combined_results AS (
                SELECT *, 
                    CASE 
                        WHEN match_type = 'exact' THEN 1.0 / (rank + 60) * 1.0  -- Highest weight
                        WHEN match_type = 'fuzzy' THEN 1.0 / (rank + 60) * 0.8  -- Second highest
                        WHEN match_type = 'keyword' THEN 1.0 / (rank + 60) * 0.6 -- Third highest
                        WHEN match_type = 'vector' THEN 1.0 / (rank + 60) * 0.4  -- Lowest weight
                    END AS rrf_score
                FROM (
                    SELECT * FROM exact_match
                    UNION ALL
                    SELECT * FROM fuzzy_match
                    UNION ALL
                    SELECT * FROM keyword_match
                    UNION ALL
                    SELECT * FROM vector_match
                ) AS all_matches
            )
            SELECT 
                conceptid,
                conceptid_name,
                toplevelhierarchy_name,
                fhir_resource,
                original_score,
                match_type,
                SUM(rrf_score) AS final_score
            FROM combined_results
            GROUP BY conceptid, conceptid_name, toplevelhierarchy_name, fhir_resource, original_score, match_type
            ORDER BY final_score DESC
            LIMIT %s
            """
            
            # Parameters for the query placeholders
            keywords_for_tsquery = ent.replace(' ', ' & ')
            
            params = (
                ent,                 # Exact match
                ent,                 # Fuzzy word match query
                ent,                 # Fuzzy word match ranking
                ent,                 # Fuzzy word match filter
                keywords_for_tsquery,  # Keyword full-text search query
                keywords_for_tsquery,  # Keyword ranking 
                keywords_for_tsquery,  # Keyword filter
                vector_str,          # Vector comparison
                vector_str,          # Vector ranking
                10                   # Results limit (top 10)
            )
            
            try:
                cur.execute(query, params)
                rows = cur.fetchall()
            except Exception as e:
                print(f"Database query error for entity '{ent}': {str(e)}")
                rows = []
            
            # Store matches
            top_matches = []
            for row in rows:
                if row:  # Ensure row exists
                    row_dict = dict(row)
                    match_item = {
                        'name': row_dict.get('conceptid_name', '').split(',')[0],
                        'code': row_dict.get('conceptid', ''),
                        'match_type': row_dict.get('match_type', ''),
                        'score': round(float(row_dict.get('final_score', 0)), 3)
                    }
                    top_matches.append(match_item)
            
            # If no rows were found, provide default values
            if not rows:
                fhir_resource = "Observation"
                sct_name = ""
                sct_id = ""
                tlh_name = ""
            else:
                # Keep the same output format as the original code
                first_row = dict(rows[0]) if rows else {}
                fhir_resource = first_row.get('fhir_resource', 'Observation')
                sct_name = first_row.get('conceptid_name', '').split(',')[0]
                sct_id = first_row.get('conceptid', '')
                tlh_name = first_row.get('toplevelhierarchy_name', '').split(',')[0]
                
                # Handle special cases
                if sct_name == 'Gender':
                    fhir_resource = 'Patient.Gender'
                if 'age ' in sct_name.lower():
                    fhir_resource = 'Patient.Age'
            
            # Add this entity's matches to the column's matches array
            column_top_matches[col_name] = top_matches
            
            col_cds.append({
                'Entity': ent,
                'SCT_Name': sct_name,
                'SCT_ID': sct_id,
                'TLH_Name': tlh_name,
                'FHIR_Resource': fhir_resource,
                'Top_Matches': top_matches
            })
            
        snmd_cds.append({col_name: col_cds})
    
    # Preserve the original post-processing
    snmd_cds = mark_condition_resources(snmd_cds)
    snmd_cds = process_snmd_cds(snmd_cds)
    
    return (snmd_cds, column_top_matches)

def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def create_col_abbrs(adarv_snmd_cds, snmd_cds):
    """
    Create abbreviated column names using SNOMED names and FHIR resource type.
    Format: first_three_of_FHIR_resource _ first_three_of_entity1_SNOMED _ first_three_of_entity2_SNOMED ...
    
    For multi-entity columns, the abbreviation is calculated after all entities have been mapped to SNOMED.
    """
    abbr_col_names = {}
    duplicate_col_check = []
    duplicate_counter = 1
    
    # First, handle ADARV mappings
    for col in adarv_snmd_cds:
        orig_col_name = col['Original_Column_Name']
        
        # Skip if we don't have entities with SCT_Name
        if not col.get('Entities'):
            continue
            
        # Extract FHIR resource prefix (first 3 letters)
        fhir_prefix = col['FHIR_Resource'][:3].lower()
        
        # Extract SNOMED terms for each entity
        snomed_prefixes = []
        for entity in col['Entities']:
            if 'SCT_Name' in entity and entity['SCT_Name']:
                # Use first 3 characters of SNOMED name
                snomed_name = entity['SCT_Name'].strip()
                if snomed_name:
                    snomed_prefixes.append(snomed_name[:3].lower())
        
        # Create abbreviated column name
        if snomed_prefixes:
            abbr_col = f"{fhir_prefix}_{'_'.join(snomed_prefixes)}"
            
            # Handle duplicates
            if abbr_col in duplicate_col_check:
                abbr_col = f"{abbr_col}{duplicate_counter}"
                duplicate_counter += 1
                
            abbr_col_names[orig_col_name] = abbr_col
            duplicate_col_check.append(abbr_col)
    
    # Then handle entries from snmd_cds
    for col_dict in snmd_cds:
        col_name = list(col_dict.keys())[0]
        
        # Skip if already processed from ADARV mappings
        if col_name in abbr_col_names:
            continue
        
        try:
            # Get FHIR resource prefix
            fhir_prefix = col_dict.get('FHIR_Resource', 'obs')[:3].lower()
            
            # Get SNOMED names for entities
            snomed_prefixes = []
            for entity in col_dict[col_name]:
                if isinstance(entity, dict) and 'SCT_Name' in entity and entity['SCT_Name']:
                    snomed_name = entity['SCT_Name'].strip()
                    if snomed_name:
                        snomed_prefixes.append(snomed_name[:3].lower())
            
            # Create abbreviated column name
            if snomed_prefixes:
                abbr_col = f"{fhir_prefix}_{'_'.join(snomed_prefixes)}"
                
                # Handle duplicates
                if abbr_col in duplicate_col_check:
                    abbr_col = f"{abbr_col}{duplicate_counter}"
                    duplicate_counter += 1
                    
                abbr_col_names[col_name] = abbr_col
                duplicate_col_check.append(abbr_col)
            else:
                # Fallback if no SNOMED names available
                abbr_col = f"{fhir_prefix}_{remove_punctuation(col_name)[:3].lower()}"
                if abbr_col in duplicate_col_check:
                    abbr_col = f"{abbr_col}{duplicate_counter}"
                    duplicate_counter += 1
                abbr_col_names[col_name] = abbr_col
                duplicate_col_check.append(abbr_col)
                
        except Exception as e:
            print(f"Error creating abbreviation for {col_name}: {e}")
            # Fallback to simple abbreviation
            abbr_col = f"col_{len(abbr_col_names)}"
            abbr_col_names[col_name] = abbr_col
    
    return abbr_col_names

def create_valueset(df):
    for col in df.columns:
        # Initialize attrs dictionary if it doesn't exist
        if not hasattr(df[col], 'attrs'):
            df[col].attrs = {}
            
        # if df[col].attrs["FHIR_Resource"] == "condition":
        #     df[col].attrs['valueSet'] = 'valueBoolean'
        # elif df[col].attrs["FHIR_Resource"] == "observation":
        unique_values = df[col].dropna().unique()
        
        if pd.api.types.is_bool_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueBoolean'
        elif pd.api.types.is_string_dtype(df[col].dropna()):
            if set(unique_values).issubset({'yes', 'no', 'Yes', 'No', 'YES', 'NO'}):
                df[col].attrs['valueSet'] = 'valueBoolean'
            else:
                df[col].attrs['valueSet'] = 'valueString'
        elif pd.api.types.is_datetime64_any_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueDateTime'
        elif pd.api.types.is_integer_dtype(df[col].dropna()) or pd.api.types.is_float_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueInteger'
        else:
            df[col].attrs['valueSet'] = 'valueString'
    return df

def check_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)
    return result

def value_modifier(df):
    for col in df.columns:
        if df[col].attrs['FHIR_Resource'] == 'observation' and df[col].attrs['valueSet'] == 'valueBoolean':
            value_modifier = {}
            unique_values = df[col].unique()
            for value in unique_values:
                value = str(value)
                if not pd.isna(value):
                    sentiment = check_sentiment(str(value))
                    if sentiment[0]['label'] == 'POSITIVE':
                        value_modifier[value] = True
                    else:
                        value_modifier[value] = False
                else:
                    value_modifier[value] = False
            df[col].attrs['valueModifier'] = value_modifier
        elif df[col].attrs['FHIR_Resource'] == 'condition':
            value_modifier = {}
            unique_values = df[col].unique()
            for value in unique_values:
                value = str(value)
                if not pd.isna(value):
                    sentiment = check_sentiment(str(value))
                    if sentiment[0]['label'] == 'POSITIVE':
                        value_modifier[value] = 'confirmed'
                    else:
                        value_modifier[value] = 'refuted'
                else:
                    value_modifier[value] = 'refuted'
            df[col].attrs['valueModifier'] = value_modifier
    return df



class Patient:
    def __init__(self):
        self._id = ""
        self._gender = ""

    def set_id(self, idd):
        self._id = str(idd)

    def set_gender(self, gender):
        if type(gender) == str:
            if (gender.lower() == 'male'):
                self._gender = "male"
            if (gender.lower() == 'female'):
                self._gender = "female"
        else:
            if (gender == 1):
                self._gender = "male"
            if (gender == 2):
                self._gender = "female"

    def create_patient(self, patient_template):
        pat = copy.deepcopy(patient_template)
        pat["id"] = self._id
        pat["gender"] = self._gender
        return pat

def create_condition(rid, idd, snm_cd, valset, val, cond_template, valueMap=None):
    coding_tmplt =  {"code": "", "display":"", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(cond_template)
    cond["id"] = rid
    cond["subject"]["reference"] = "Patient/" + str(idd)
    
    # Make sure snm_cd is a list and handle empty cases
    if not snm_cd or len(snm_cd) == 0:
        # Default coding for empty entities
        ct = copy.deepcopy(coding_tmplt)
        ct["code"] = "261665006"  # Unknown (qualifier value)
        ct["display"] = "Unknown"
        cond["code"]["coding"] = [ct]
    else:
        cond["code"]["coding"] = []
        for i in snm_cd:
            try:
                ct = copy.deepcopy(coding_tmplt)
                # Handle different formats of SCT_ID
                if isinstance(i, dict) and 'SCT_ID' in i:
                    if isinstance(i['SCT_ID'], list) and i['SCT_ID']:
                        ct["code"] = str(i['SCT_ID'][0])
                    else:
                        ct["code"] = str(i['SCT_ID'])
                    
                    # Safely handle SCT_Name
                    if 'SCT_Name' in i and i['SCT_Name']:
                        ct["display"] = i['SCT_Name']
                    else:
                        ct["display"] = "Unspecified"
                else:
                    # Unknown format, use defaults
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
                
                cond["code"]["coding"].append(ct)
            except Exception as e:
                print(f"Error processing SNOMED entity: {e}, {i}")
                # Add default if there's an error
                ct = copy.deepcopy(coding_tmplt)
                ct["code"] = "261665006"
                ct["display"] = "Error processing entity"
                cond["code"]["coding"].append(ct)
    
    # Set verification status based on value and mapping
    if not pd.isna(val) and valueMap and str(val) in valueMap:
        cond["verificationStatus"]["coding"][0]["code"] = valueMap[str(val)]
    else:
        # Default based on value truthiness
        if val and not pd.isna(val):
            cond["verificationStatus"]["coding"][0]["code"] = "confirmed"
        else:
            cond["verificationStatus"]["coding"][0]["code"] = "refuted"
            
    return cond

def create_obsv(rid, idd, snm_cd, valset, val, obs_template, valueMap=None):
    coding_tmplt =  {"code": "", "display":"", "system": "http://snomed.info/sct/"}
    obs = copy.deepcopy(obs_template)
    obs["id"] = rid
    obs["subject"]["reference"] = "Patient/" + str(idd)
    
    # Make sure snm_cd is a list and handle empty cases
    if not snm_cd or len(snm_cd) == 0:
        # Default coding for empty entities
        ct = copy.deepcopy(coding_tmplt)
        ct["code"] = "261665006"  # Unknown (qualifier value)
        ct["display"] = "Unknown"
        obs["code"]["coding"] = [ct]
    else:
        obs["code"]["coding"] = []
        for i in snm_cd:
            try:
                ct = copy.deepcopy(coding_tmplt)
                # Handle different formats of SCT_ID
                if isinstance(i, dict) and 'SCT_ID' in i:
                    if isinstance(i['SCT_ID'], list) and i['SCT_ID']:
                        ct["code"] = str(i['SCT_ID'][0])
                    else:
                        ct["code"] = str(i['SCT_ID'])
                    
                    # Safely handle SCT_Name
                    if 'SCT_Name' in i and i['SCT_Name']:
                        ct["display"] = i['SCT_Name']
                    else:
                        ct["display"] = "Unspecified"
                else:
                    # Unknown format, use defaults
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
                
                obs["code"]["coding"].append(ct)
            except Exception as e:
                print(f"Error processing SNOMED entity: {e}, {i}")
                # Add default if there's an error
                ct = copy.deepcopy(coding_tmplt)
                ct["code"] = "261665006"
                ct["display"] = "Error processing entity"
                obs["code"]["coding"].append(ct)

    # Process value mapping if provided
    if valueMap is not None:
        if "valueInteger" in valset:
            val_str = str(val)
            if val_str in valueMap:
                obs["valueInteger"] = valueMap[val_str]
            else:
                # Default fallback
                obs["valueInteger"] = 0
        elif "valueBoolean" in valset:
            val_str = str(val).replace(".0", "")
            if val_str in valueMap:
                obs["valueBoolean"] = valueMap[val_str]
            else:
                # Default fallback
                obs["valueBoolean"] = False
        elif "valueDateTime" in valset:
            if val in valueMap:
                obs["valueDateTime"] = valueMap[val]
            else:
                # Default fallback
                obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return obs

    # Handle different value types with sensible defaults
    if "valueInteger" in valset:
        if pd.isna(val):
            obs["valueInteger"] = 0
        else:
            try:
                obs["valueInteger"] = int(float(val))
            except (ValueError, TypeError):
                obs["valueInteger"] = 0
    elif "valueBoolean" in valset:
        if pd.isna(val):
            obs["valueBoolean"] = False
        elif val == 0 or val == "0" or val == "0.0" or val == "False" or val == "false" or val == "no" or val == "No":
            obs["valueBoolean"] = False
        else:
            obs["valueBoolean"] = True
    elif "valueDateTime" in valset:
        if pd.isna(val):
            obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        else:
            try:
                if isinstance(val, (datetime.datetime, pd.Timestamp)):
                    obs["valueDateTime"] = val.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    # Try to parse string to datetime
                    dt = pd.to_datetime(val)
                    obs["valueDateTime"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except:
                # Fallback to current time if parsing fails
                obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    elif "valueString" in valset:
        if pd.isna(val):
            obs["valueString"] = ""
        else:
            obs["valueString"] = str(val)
    else:
        # If we can't determine the value type, default to string
        if pd.isna(val):
            obs["valueString"] = ""
        else:
            obs["valueString"] = str(val)
    
    return obs

def create_location(rid, val, loc_template):
    coding_tmplt =  {"code": "", "display":"", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(loc_template)
    cond["id"] = rid
    cond["name"] = val["location name"]
    cond["position"]["longitude"] = val["longitude"]
    cond["position"]["latitude"] = val["latitude"]
    return cond

def create_group(rid, patient_ids, grp_template):
    grp = copy.deepcopy(grp_template)
    grp["id"] = rid
    grp["name"] = rid
    grp["title"] = rid
    grp["status"] = "active"
    grp["type"] = "person"
    grp["publisher"] = "ICMR"
    grp["membership"] = "enumerated"
    grp["member"] = patient_ids
    grp["quantity"] = len(patient_ids)
    return grp

def create_Encounter(rid, idd, snm_cd, valset, val, lids, enc_template):
    coding_tmplt =  {"coding": "", "display":"", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(enc_template)
    cond["id"] = rid
    cond["subject"]["reference"] = "Patient/" + str(idd)
    cond["plannedStartDate"] = str(val)
    for lid in lids:
        dct = {"location": {"reference": ""}}
        dct["location"]["reference"] = f"Location/{lid}"
        cond["location"].append(dct)
    return cond

def create_intake(rid, idd, snm_cd, additionalfhir, valset, val, text, obs_template, valueMap=None):
    coding_tmplt =  {"code": "", "display":"", "system": "http://snomed.info/sct/"}
    timing_tmplt = {"repeat": {"when": ["MORN"]}}
    ni = copy.deepcopy(obs_template)
    ni["id"] = rid
    ni["subject"]["reference"] = "Patient/" + str(idd)
    
    # Handle string vs. list format for snm_cd and safely parse
    try:
        if isinstance(snm_cd, str):
            concepts = snm_cd.split(",")
        elif isinstance(snm_cd, list):
            concepts = snm_cd
        else:
            # Default concept if format is unexpected
            concepts = [{"SCT_ID": "261665006", "SCT_Name": "Unknown"}]
    except:
        concepts = [{"SCT_ID": "261665006", "SCT_Name": "Unknown"}]
    
    ni["code"]["coding"] = []
    
    # Process concepts based on format
    for concept in concepts:
        try:
            ct = copy.deepcopy(coding_tmplt)
            
            if isinstance(concept, str) and "(" in concept:
                # Parse format like "Name (Code)"
                parts = concept.split("(")
                ct["display"] = parts[0].strip()
                ct["code"] = parts[1].strip(")").strip()
            elif isinstance(concept, dict):
                # Handle dictionary format
                if 'SCT_ID' in concept:
                    if isinstance(concept['SCT_ID'], list):
                        ct["code"] = str(concept['SCT_ID'][0])
                    else:
                        ct["code"] = str(concept['SCT_ID'])
                    
                    if 'SCT_Name' in concept:
                        ct["display"] = concept['SCT_Name']
                    else:
                        ct["display"] = "Unknown"
                else:
                    # Default if no SCT_ID
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
            else:
                # Default for unsupported format
                ct["code"] = "261665006"
                ct["display"] = "Unknown"
                
            ni["code"]["coding"].append(ct)
        except Exception as e:
            print(f"Error processing concept in create_intake: {e}")
            # Add default if error
            ct = copy.deepcopy(coding_tmplt)
            ct["code"] = "261665006"
            ct["display"] = "Error"
            ni["code"]["coding"].append(ct)

    # Set text if provided
    if text and isinstance(text, str):
        ni["code"]["text"] = text

    # Process additional FHIR properties
    if additionalfhir and not isinstance(additionalfhir, float):
        # Apply timing info
        ni["effectiveTiming"] = timing_tmplt
        
        # Convert to string to handle various input types
        additionalfhir_str = str(additionalfhir)
        
        if "MORN" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "MORN"
        elif "AFT" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "AFT"
        elif "EVE" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "EVE"
        elif "NIGHT" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "NIGHT"

    # Apply value mapping if provided
    if valueMap is not None:
        val_str = str(val).replace(".0", "") if not pd.isna(val) else ""
        
        if "valueInteger" in valset and val_str in valueMap:
            ni["valueInteger"] = valueMap[val_str]
        elif "valueBoolean" in valset and val_str in valueMap:
            ni["valueBoolean"] = valueMap[val_str]
        elif "valueDateTime" in valset and str(val) != "NaT" and val in valueMap:
            ni["valueDateTime"] = valueMap[val]
        else:
            # Set default value type for mapped but not matched value
            if "valueInteger" in valset:
                ni["valueInteger"] = 0
            elif "valueBoolean" in valset:
                ni["valueBoolean"] = False
            elif "valueDateTime" in valset:
                ni["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            else:
                ni["valueString"] = ""
    # Process regular values without mapping
    else:
        if valset == "valueBoolean" or "valueBoolean" in valset:
            if pd.isna(val) or val == 0 or val == "0" or val == "0.0" or val == "False" or val == "false" or val == "no" or val == "No":
                ni["valueBoolean"] = False
            else:
                ni["valueBoolean"] = True
        elif valset == "valueString" or "valueString" in valset:
            ni["valueString"] = str(val) if not pd.isna(val) else ""
        elif valset == "valueInteger" or "valueInteger" in valset:
            try:
                ni["valueInteger"] = int(float(val)) if not pd.isna(val) else 0
            except (ValueError, TypeError):
                ni["valueInteger"] = 0
        elif valset == "valueDateTime" or "valueDateTime" in valset:
            try:
                if pd.isna(val):
                    ni["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                elif isinstance(val, (datetime.datetime, pd.Timestamp)):
                    ni["valueDateTime"] = val.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    ni["valueDateTime"] = pd.to_datetime(val).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                print(f"Error converting datetime: {e}")
                ni["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        else:
            # Default to string for unknown value types
            ni["valueString"] = str(val) if not pd.isna(val) else ""
    
    return ni

def create_loc(idd,  val, loc_template):
    loc = copy.deepcopy(loc_template)
    loc["subject"]["reference"] = "Patient/" + str(idd)
    loc["name"] = val
    return loc

def load_fhir_data(df, dataset_name):
    patient_template, age_template, obs_template, loc_template, cond_template, ni_template, enc_template, grp_template = load_templates()
    fullbundle = []
    baseurl = "https://adarv.icmr.gov.in/add_mtzion/base/"
    location_data = []
    patient_ids = []
    gndr = None
    
    # Safely find gender column
    for col in df.columns:
        try:
            # Use getattr with default empty dict to avoid KeyError
            attrs = getattr(df[col], 'attrs', {})
            if 'FHIR_Resource' in attrs and attrs['FHIR_Resource'] == 'Patient.Gender':
                gndr = col
                break
        except Exception as e:
            print(f"Error checking column {col} for gender: {e}")
    
    sample_bundle = []
    bundle_count = 1
    for index, data in df.iterrows():
        bundle = {"resourceType": "Bundle", "type": "transaction", "entry": []}
        resource_template = {"request": {"method": "PUT"}, "fullUrl": baseurl, "resource": {}}
        pat = Patient()
        patid = lid = str(uuid.uuid1())
        pat.set_id(patid)
        patient_ids.append({
                "entity": {"reference" : f"Patient/{patid}"}
            })
        if gndr:
            pat.set_gender(data[gndr])
        rpt = copy.deepcopy(resource_template)
        rpt["fullUrl"] = baseurl+"Patient/" + pat._id
        rpt["resource"] = pat.create_patient(patient_template)
        bundle["entry"].append(rpt)

        for k in df.columns:
            rpt = copy.deepcopy(resource_template)
            rid = str(uuid.uuid1())
            try:
                # Safe attribute access with defaults
                attrs = getattr(df[k], 'attrs', {})
                fhir_resource = attrs.get('FHIR_Resource', 'observation')
                entities = attrs.get('Entities', [])
                value_set = attrs.get('valueSet', 'valueString')
                val_mod = attrs.get('valueModifier', None)
                
                # Skip Gender as it's handled separately
                if fhir_resource == 'Patient.Gender':
                    continue

                if fhir_resource == 'condition':
                    cd = create_condition(rid, pat._id, entities, value_set, data[k],
                                        cond_template, val_mod)
                    rpt["fullUrl"] = baseurl+"Condition/" + rid
                    rpt["resource"] = cd
                    bundle["entry"].append(rpt)
    
                elif fhir_resource == 'observation': 
                    obs = create_obsv(rid, pat._id, entities, value_set, data[k],
                                    obs_template, val_mod)
                    if obs is None:
                        continue
                    rpt["fullUrl"] = baseurl+ "Observation/" + rid
                    rpt["resource"] = obs
                    bundle["entry"].append(rpt)

                elif fhir_resource == 'observation (intake)':
                    additional_props = attrs.get('Additional FHIR Properties', "")
                    itk = create_intake(rid, pat._id, entities, additional_props,
                                    value_set, data[k], k, obs_template, val_mod)
                    rpt["fullUrl"] = baseurl+"Observation/" + rid
                    rpt["resource"] = itk
                    bundle["entry"].append(rpt)
    
            except Exception as e:
                print(f"Error processing column {k}:")
                print(e)
                print(traceback.format_exc())
                continue
    
        for entry in bundle["entry"]:
            resource_data = entry["resource"]
            url = f"http://65.0.127.208:30007/fhir/{resource_data['resourceType']}/{resource_data['id']}"
            # r = requests.put(url,
            #             headers= {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"},
            #             data=json.dumps(resource_data))
            # print(r.status_code)
            # print(r.json())
    
        fullbundle.append(bundle)
        if bundle_count == 1:
            sample_bundle.append(bundle)
            bundle_count += 1

    group_id = str(dataset_name)
    url = f"http://65.0.127.208:30007/fhir/Group/{group_id}"
    resource_data = create_group(group_id, patient_ids, grp_template)

    # r = requests.put(url,
    #                 headers= {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"},
    #                 data=json.dumps(resource_data))
    # print(r.status_code)
    # print(r.json())

    with open("bundle.json", "w") as f:
        json.dump(fullbundle, f, indent=4)

    return fullbundle, sample_bundle