# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:31:57 2023

@author: manan
"""
import os
import re
from tika import parser
import spacy
import requests
from spacy.matcher import Matcher
#import json
from fastapi import FastAPI, HTTPException
import time
from urllib.parse import urlparse

app = FastAPI()
"""
nlp_model = spacy.load('../ml-models/output_6357_gpu_acc_e2e/model-best')
nlp_we_model = spacy.load('../ml-models/WE_1991_GPU-20230919T083318Z-001/WE_1991_GPU/model-best')
nlp_education_model = spacy.load('../ml-models/education_output_1030_gpu_specialization-20230918T054413Z-001/education_output_1030_gpu_specialization/model-best')
nlp = spacy.load('en_core_web_lg')
"""
nlp_model = spacy.load('Resume_15622_GPU/model-best')
nlp_we_model = spacy.load('WE_2496_GPU/model-best')
nlp_education_model = spacy.load('Ed_2255_7_GPU/model-best')
nlp = spacy.load('en_core_web_lg')


# st = time.time()
# nlp_model = spacy.load(r'C:\Users\manan\Documents\Trials - Auto-fill API\API_for_deployment\ml_models\output_6357_gpu_acc_e2e\output_6357_gpu_acc_e2e\model-best')
# en = time.time()
# print(en - st)
# print("Model 1 loaded")
# nlp_we_model = spacy.load(r'C:\Users\manan\Documents\Trials - Auto-fill API\API_for_deployment\ml_models\WE_1991_GPU-20230919T083318Z-001\WE_1991_GPU\model-best')
# en2 = time.time()
# print(en2 - en)
# print("Model 2 loaded")
# nlp_education_model = spacy.load(r'C:\Users\manan\Documents\Trials - Auto-fill API\API_for_deployment\ml_models\education_output_1030_gpu_specialization-20230918T054413Z-001\education_output_1030_gpu_specialization\model-best')
# en3 = time.time()
# print(en3 - en2)
# print("Model 3 loaded")
# nlp = spacy.load('en_core_web_lg')
# en4 = time.time()
# print(en4 - en3)
# print("Model 4 loaded")

# Initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

def extract_text_with_tika(pdf_path):
    """
    Extracts text content from a given PDF file.

    Parameters:
    - pdf_path (str): The path to the PDF file.

    Returns:
    - str: Extracted text content from the PDF.
    """

    parsed_pdf = parser.from_file(pdf_path)
    return parsed_pdf['content'].strip()

def extract_text_from_raw_pdf_tika(pdf_content):
    """
    Extract text from raw PDF content using Apache Tika.

    Parameters:
    - pdf_content: Raw PDF content as bytes.

    Returns:
    - Extracted text from the PDF.
    """
    
    # Write the PDF content to a temporary file because Tika's parser requires a file path
    with open("temp.pdf", "wb") as f:
        f.write(pdf_content)
    
    # Use Tika's parser to extract text
    result = parser.from_file("temp.pdf")
    extracted_text = result.get("content", "")
    
    # Clean up the temporary file
    os.remove("temp.pdf")
    
    return extracted_text.strip()


def extract_resume_entities(resume_text, nlp_model):
  # Process the text with the trained model
      doc = nlp_model(resume_text)
      
      resume_json = {}
      
      for ent in doc.ents:
          resume_json[ent.label_] = ent.text        
      return resume_json

def fetch_resume_data(resume_url):
    try:
        response = requests.get(resume_url)
        if response.status_code == 200:
            resume_data = extract_text_from_raw_pdf_tika(response.content)
            #resume_data = response.text
            return resume_data
        else:
            print(f"Error: Unable to fetch resume data. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def process_skills_for_autofill(resume_json):
    
      data_skills = {}
      
      try:
          skills_string = resume_json['SKILLS']
      except KeyError:
          skills_string = ''
          data_skills = {'skills' : []}
          return data_skills
      
      # Split the string by the bullet point and then further split by commas
      lines = [line.strip() for line in skills_string.split("•") if line.strip() and not line.startswith("Skills")]
  
      skills_list = []
      for line in lines:
          # Check if line has a colon, then split by the colon. Else, use the whole line.
          items = line.split(":")[-1].split(",") if ":" in line else line.split(",")
          
          # Strip whitespace and add to skills_list
          for item in items:
              item = item.replace('\n', '')
              skills_list.append(item.strip())
  
      resume_json['SKILLS'] = skills_list
      
      data_skills['skills'] = resume_json['SKILLS']
  
      return data_skills

def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern], on_match=None)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def get_location(resume_text):
    nlp_text = nlp(resume_text)
    
    # Extract GPE entities
    locations = [ent.text.strip().replace('\n', '') for ent in nlp_text.ents if ent.label_ == 'GPE']

    # If there are comma-separated locations
    if len(locations) > 1 and ',' in resume_text:
        for loc in locations:
            if loc in resume_text:
                comma_index = resume_text.find(',', resume_text.find(loc))
                if comma_index != -1 and resume_text[comma_index+1:comma_index+2].isspace():
                    return loc
                # if there's no comma after the location or if the next character after comma isn't space, it's probably part of some other context. So, continue the search
            else:
                continue
    
    # If there's only a single GPE or none
    return locations[0] if locations else ''
    
def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    finds = r.findall(string)
    if finds == []:
        return ['']
    else:
        return finds

def get_cid_and_numbers(string):
    # First, remove URLs from the string
    url_pattern = r"https?://[^\s]+"
    string_without_urls = re.sub(url_pattern, "", string)

    # Regular expression pattern for phone numbers with international codes
    pattern = r'((\+\d{1,3})\s?(\d{7,10})|(\d{7,10}))'
    
    # Find all phone numbers in the string using the pattern
    matches = re.findall(pattern, string_without_urls)
    
    cids = []
    phone_numbers = []

    for match in matches:
        # If the second tuple element (CID) is not empty, it has a CID
        if match[1]:
            cids.append(match[1])
            phone_numbers.append(match[2])
        else:  # This means it's the format without the CID
            cids.append(None)
            phone_numbers.append(match[3])
    
    # Provide default values if no matches were found
    if not cids:
        cids = ['']
    if not phone_numbers:
        phone_numbers = ['']

    return cids, phone_numbers

def process_personal_details_for_autofill(resume_json):

    try:
        text = resume_json['PERSONAL DETAILS']
    except KeyError:
        text = ''
    
    # Define regex patterns
    links_pattern = r"https?://[^\s]+"
    
    # Extract information
    links_matches = list(set(re.findall(links_pattern, text)))
    
    # Extract and format domain names for each link
    portfolio_links = []
    for link in links_matches:
        domain = urlparse(link).netloc
        if "www." in domain:
            company_name = domain.split("www.")[-1].split(".")[0]
        else:
            company_name = domain.split(".")[0]
        portfolio_links.append({"company_name": company_name.replace('\n', ''), "URL": link.replace('\n', '')})
    
    # Fill in the JSON
    data_pd = {
        # "FirstName": extract_name(text).split()[0],
        # "LastName": extract_name(text).split()[1],
        # "Email address": get_email_addresses(text)[0],
        "cid": get_cid_and_numbers(text)[0][0],
        "phone_number": get_cid_and_numbers(text)[1][0],
        "location": get_location(text).replace('-', ' '),
        "portfolio_links": portfolio_links
    }
    
    print(data_pd)
    
    # Return the updated JSON
    return data_pd


def process_work_experience_for_autofill(resume_json):
    
    try:
        we_string = resume_json['WORK EXPERIENCE']
    except KeyError:
        we_string = ''
        
    we_doc = nlp_we_model(we_string)

    work_experience_list = []
    current_work_experience = {}
    encountered_jobs = set()

    for ent in we_doc.ents:
        text = ent.text.strip()
        label = ent.label_

        if label == 'JOB_TITLE':
            if "job_title" in current_work_experience:  # If we encounter a new job title, store the previous work experience
                work_experience_list.append(current_work_experience)
                current_work_experience = {}
            current_work_experience["job_title"] = text.replace('\n', '')
            
        elif label == 'COMPANY_NAME':
            current_work_experience["company"] = text.replace('\n', '')
            
        elif label == 'START_DATE':
            current_work_experience["start_date"] = text.replace('\n', '')
        elif label == 'END_DATE':
            current_work_experience["end_date"] = text.replace('\n', '')
        elif label == 'LOCATION':
            location_text = text.replace('\n', ' ')
            location_text = location_text.replace('-', ' ')
            try:
                current_work_experience["location"] = location_text.strip().split(',')[0]
            except Exception as e:
                print(e)
                current_work_experience["location"] = location_text
        elif label == 'ROLES_&_RESPONSIBILITIES':
            current_work_experience["roles"] = text.replace('\n', '')
        elif label == 'SKILLS':
            current_work_experience["skills"] = text.replace('\n', '')

    # To capture the last work experience, if it wasn't added yet
    if current_work_experience and current_work_experience not in work_experience_list:
        work_experience_list.append(current_work_experience)

    return {"work_experience": work_experience_list}


def process_education_for_autofill(resume_json):
    try:
        education_string = resume_json['EDUCATION']
    except KeyError:
        education_string = ''

    education_doc = nlp_education_model(education_string)

    # Initialize a dictionary to store grouped entities
    grouped_data = {}

    # Iterate over all the entities extracted from the resume
    for ent in education_doc.ents:
        text = ent.text.strip().replace('\n', ' ')
        label_parts = ent.label_.split(' - ')

        category = label_parts[0]  # e.g., 'DEGREE', 'START YEAR'
        degree_name = label_parts[1]  # e.g., 'BACHELORS', 'MASTERS'

        # If this degree name isn't in our grouped_data yet, add it
        if degree_name not in grouped_data:
            grouped_data[degree_name] = {
                "qualification_level":"",
                "qualification": "",
                "institute": "",
                "start_date": "",
                "end_date": "",
                "field_of_study": "",
                "percentage_or_cgpa": ""  # Added as an example; you can add more fields as necessary
            }
            
        if degree_name == '10TH':
            level_name = 'matriculation'
        elif degree_name == '12TH':
            level_name = 'intermediate'
        elif degree_name == 'DIPLOMA':
            level_name = 'diploma'
        elif degree_name == 'BACHELORS':
            level_name = 'graduate'
        elif degree_name == 'MASTERS':
            level_name = 'post_graduate'
        elif degree_name == 'PH. D.' or degree_name == 'PH.D.':
            level_name = 'doctorate'
        else:
            level_name = ''

        # Fill in the information based on the category
        if category == "DEGREE":
            grouped_data[degree_name]["qualification_level"] = level_name
            grouped_data[degree_name]["qualification"] = text
        elif category == "INSTITUTION":
            grouped_data[degree_name]["institute"] = text
        elif category == "START YEAR":
            grouped_data[degree_name]["start_date"] = text
        elif category == "END YEAR":
            grouped_data[degree_name]["end_date"] = text
        elif category == "FIELD OF STUDY":
            grouped_data[degree_name]["field_of_study"] = text
        elif category == "PERCENTAGE":
            grouped_data[degree_name]["percentage_or_cgpa"] = text
        # Add more conditions here if you have other entities, like "MARKS"

    # Transform the grouped entities into the desired JSON format
    education_details_list = []

    for _, data in grouped_data.items():
        education_detail = {
            "qualification_level": data["qualification_level"],  # Can adjust this if necessary
            "qualification": data["qualification"],
            "field_of_study": data["field_of_study"],
            "course_type": "fulltime",
            "institute": data["institute"],
            "start_date": data["start_date"],
            "end_date": data["end_date"],
            "description": "",  # This data point doesn't seem to be extracted by your entities
            # "measuring_type": "Percentage",  # Default value
            # "out_of": 100,  # Default value
            "percentage_or_cgpa": data["percentage_or_cgpa"]
        }
        education_details_list.append(education_detail)

    return {"education_details": education_details_list}


def process_profile_summary_for_autofill(resume_json):
    try:
        intro_data = {"introduction" : resume_json['PROFILE SUMMARY'].replace('\n', '')}
    except:
        intro_data = {"introduction" : ''}
    return intro_data

@app.get("/parse_resume")
async def parse_resume(link: str):
    # Fetch the resume content
    st2 = time.time()
    resume_text = fetch_resume_data(link)
    if not resume_text:
        raise HTTPException(status_code=404, detail="Unable to fetch the resume content")

    # Process the resume content
    processed_data = await process_resume(resume_text)
    en_1 = time.time()
    print(en_1 - st2)
    return processed_data

async def process_resume(resume_text: str) -> dict:
    # Extract and process entities from the resume
    resume_json = extract_resume_entities(resume_text, nlp_model)
    skills_resume_json = process_skills_for_autofill(resume_json)
    pd_resume_json = process_personal_details_for_autofill(resume_json)
    we_resume_json = process_work_experience_for_autofill(resume_json)
    ed_resume_json = process_education_for_autofill(resume_json)
    intro_resume_json = process_profile_summary_for_autofill(resume_json)

    # Merging dictionaries
    combined_dict = {
        **pd_resume_json,
        **intro_resume_json,
        **ed_resume_json,
        **we_resume_json,
        **skills_resume_json
    }
    return combined_dict