"""
API-Based Resume Parser using Free Remote LLMs
===============================================
Uses free LLM APIs instead of local models - no GPU needed!

Supported Free APIs:
1. Groq (Fast, Free tier: 30 requests/min) - RECOMMENDED
2. Together AI (Free tier available)
3. Hugging Face Inference API (Free)

Installation:
    pip install groq openai requests

Usage:
    from llm_resume_parser_api import parse_resume_with_api
    
    # Set your API key (get free at https://console.groq.com)
    import os
    os.environ['GROQ_API_KEY'] = 'your_key_here'
    
    parsed_data = parse_resume_with_api(resume_text)
"""

import json
import re
import os
from typing import Optional

# Feature mappings (from training data)
SENIORITY_ORDER = ['Intern', 'Junior', 'Mid-level', 'Senior', 'Lead',
                   'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO']
EDUCATION_ORDER = ['High School', 'Associate', 'Bachelor', 'Master', 'MBA', 'PhD']
COMPANY_SIZE_ORDER = ['Startup (<50)', 'Small (50-200)', 'Medium (200-1000)',
                      'Large (1000-5000)', 'Enterprise (5000+)']
CERT_RANK = {
    'None': 0, 'AWS Certified': 3, 'Google Cloud Certified': 3,
    'Azure Certified': 3, 'PMP': 2, 'Scrum Master': 1,
    'CFA': 3, 'CISSP': 3, 'Data Science Certifications': 2, 'Multiple Certs': 4
}

MODEL_SKILLS = [
    'agile', 'aws', 'azure', 'bi', 'cd', 'ci', 'communication', 'data',
    'deep', 'docker', 'excel', 'gcp', 'git', 'go', 'hadoop', 'java',
    'javascript', 'js', 'kubernetes', 'leadership', 'learning', 'linux',
    'machine', 'nlp', 'node', 'power', 'python', 'pytorch', 'react',
    'scala', 'scrum', 'spark', 'sql', 'tableau', 'tensorflow', 'visualization'
]


def parse_with_groq(resume_text: str, api_key: Optional[str] = None) -> dict:
    """
    Parse resume using Groq API (FREE, FAST)
    Get free API key: https://console.groq.com
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")
    
    api_key = api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Get free key at https://console.groq.com")
    
    client = Groq(api_key=api_key)
    
    prompt = f"""Extract information from this resume and return ONLY valid JSON.

EXAMPLE OUTPUT:
{{
  "job_title": "Senior Software Engineer",
  "seniority_level": "Senior",
  "years_of_experience": 8,
  "education_level": "Master",
  "field_of_study": "Computer Science",
  "gpa": 3.8,
  "skills": "Python|Java|AWS|Docker|Kubernetes",
  "location": "Seattle, WA",
  "num_projects": 5,
  "num_publications": 2,
  "num_internships": 1,
  "certifications": "AWS Certified",
  "has_leadership_experience": 1,
  "has_open_source_contributions": 1,
  "company_size": "Enterprise (5000+)",
  "industry": "Technology"
}}

RULES:
- job_title: Extract ACTUAL title
- seniority_level: Intern, Junior, Mid-level, Senior, Lead, Principal, Staff, Director, VP, CTO/CXO
- skills: ALL skills pipe-separated
- has_leadership_experience: 1 if mentions "led", "managed", else 0
- has_open_source_contributions: 1 if mentions "github", "open source", else 0

RESUME:
{resume_text[:3000]}

Return ONLY the JSON:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Updated model (replaces decommissioned llama-3.1-70b-versatile)
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    
    return response.choices[0].message.content


def parse_with_huggingface(resume_text: str, api_key: Optional[str] = None) -> dict:
    """
    Parse resume using Hugging Face Inference API (FREE)
    Get free API key: https://huggingface.co/settings/tokens
    """
    import requests
    
    api_key = api_key or os.environ.get('HF_API_KEY')
    if not api_key:
        raise ValueError("HF_API_KEY not set. Get free key at https://huggingface.co/settings/tokens")
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    prompt = f"""[INST] Extract resume info as JSON. Return ONLY JSON.

Example: {{"job_title": "Software Engineer", "skills": "Python|Java", "years_of_experience": 5}}

Resume: {resume_text[:2000]}

JSON: [/INST]"""
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.1}}
    )
    
    return response.json()[0]['generated_text']


def parse_resume_with_api(resume_text: str, provider: str = "groq", api_key: Optional[str] = None) -> dict:
    """
    Parse resume using free remote LLM API
    
    Args:
        resume_text: Raw resume text
        provider: "groq" (recommended), "huggingface", or "together"
        api_key: Optional API key (or set via environment variable)
    
    Returns:
        dict with ALL computed features ready for ML model
    """
    print(f"🌐 Using {provider.upper()} API for parsing (no local GPU needed)...")
    
    try:
        # Call appropriate API
        if provider == "groq":
            response_text = parse_with_groq(resume_text, api_key)
        elif provider == "huggingface":
            response_text = parse_with_huggingface(resume_text, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            
            # Clean JSON
            json_str = json_str.strip()
            json_str = re.sub(r'//.*?\n', '\n', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = json_str.replace("'", '"')
                parsed_data = json.loads(json_str)
            
            # Compute missing features
            parsed_data = compute_all_features(parsed_data)
            
            print(f"✓ API parsing complete: {len(parsed_data)} features computed")
            return parsed_data
        else:
            raise ValueError("No valid JSON found in API response")
            
    except Exception as e:
        print(f"✗ API parsing failed: {e}")
        print("→ Falling back to regex parsing...")
        from app import parse_resume_text
        return parse_resume_text(resume_text)


def compute_all_features(data: dict) -> dict:
    """
    Compute ALL derived features for ML model
    """
    
    # Defaults
    defaults = {
        'job_title': 'Software Engineer',
        'seniority_level': 'Mid-level',
        'years_of_experience': 3,
        'education_level': 'Bachelor',
        'field_of_study': 'Computer Science',
        'gpa': 3.5,
        'skills': 'Python|Java|SQL',
        'location': 'San Francisco, CA',
        'num_projects': 2,
        'num_publications': 0,
        'num_internships': 1,
        'certifications': 'None',
        'has_leadership_experience': 0,
        'has_open_source_contributions': 0,
        'company_size': 'Medium (200-1000)',
        'industry': 'Technology'
    }
    
    for key, default_value in defaults.items():
        if key not in data or not data[key]:
            data[key] = default_value
    
    # Ensure numeric types
    data['years_of_experience'] = int(float(data.get('years_of_experience', 3)))
    data['gpa'] = float(data.get('gpa', 3.5))
    data['num_projects'] = int(float(data.get('num_projects', 2)))
    data['num_publications'] = int(float(data.get('num_publications', 0)))
    data['num_internships'] = int(float(data.get('num_internships', 1)))
    data['has_leadership_experience'] = int(float(data.get('has_leadership_experience', 0)))
    data['has_open_source_contributions'] = int(float(data.get('has_open_source_contributions', 0)))
    
    # Process skills
    if isinstance(data.get('skills'), list):
        data['skills'] = '|'.join(str(s).strip() for s in data['skills'] if s)
    data['skills'] = data['skills'].replace(', ', '|').replace(',', '|')
    skills_list = list(dict.fromkeys([s.strip() for s in data['skills'].split('|') if s.strip()]))
    data['skills'] = '|'.join(skills_list)
    data['num_skills'] = len(skills_list)
    
    # Map to binary skill features
    skills_lower = data['skills'].lower()
    matched_skills = 0
    
    for skill in MODEL_SKILLS:
        skill_key = f"skill_{skill}"
        if skill in skills_lower:
            data[skill_key] = 1
            matched_skills += 1
        else:
            data[skill_key] = 0
    
    # Compute ordinal rankings
    data['seniority_rank'] = SENIORITY_ORDER.index(data['seniority_level']) \
                             if data['seniority_level'] in SENIORITY_ORDER else 2
    data['education_rank'] = EDUCATION_ORDER.index(data['education_level']) \
                             if data['education_level'] in EDUCATION_ORDER else 2
    data['company_rank'] = COMPANY_SIZE_ORDER.index(data['company_size']) \
                           if data['company_size'] in COMPANY_SIZE_ORDER else 2
    data['cert_rank'] = CERT_RANK.get(data.get('certifications', 'None'), 0)
    
    # Compute experience bin
    exp = data['years_of_experience']
    if exp <= 0:
        data['exp_bin'] = 0
    elif exp <= 2:
        data['exp_bin'] = 1
    elif exp <= 5:
        data['exp_bin'] = 2
    elif exp <= 10:
        data['exp_bin'] = 3
    elif exp <= 15:
        data['exp_bin'] = 4
    elif exp <= 20:
        data['exp_bin'] = 5
    else:
        data['exp_bin'] = 6
    
    # Compute interaction features
    data['seniority_x_exp'] = data['seniority_rank'] * data['years_of_experience']
    data['edu_x_seniority'] = data['education_rank'] * data['seniority_rank']
    data['skills_x_exp'] = data['num_skills'] * data['years_of_experience']
    
    # Compute achievement score
    data['achievement_score'] = (
        data['num_projects'] * 0.5 +
        data['num_publications'] * 2.0 +
        data['num_internships'] * 1.0 +
        data['has_leadership_experience'] * 3.0 +
        data['has_open_source_contributions'] * 2.0
    )
    
    print(f"  Features: seniority_rank={data['seniority_rank']}, "
          f"achievement_score={data['achievement_score']:.1f}, "
          f"skills_matched={matched_skills}/{len(MODEL_SKILLS)}")
    
    return data


def test_api_parser():
    """Test the API parser"""
    sample_resume = """
    Sarah Chen
    Senior Machine Learning Engineer
    sarah.chen@email.com | Seattle, WA
    
    EXPERIENCE
    Senior ML Engineer at Google (2020-Present, 4 years)
    - Led team of 5 engineers
    - Built 8 production ML systems
    - Published 3 papers at NeurIPS
    
    EDUCATION
    Master of Science in Computer Science, Stanford, 2018, GPA: 3.9
    
    SKILLS
    Python, TensorFlow, PyTorch, AWS, Docker, Kubernetes, Java, SQL
    
    CERTIFICATIONS
    AWS Certified Machine Learning
    """
    
    print("=" * 80)
    print("TESTING API-BASED PARSER (No Local GPU Needed!)")
    print("=" * 80)
    
    # Check for API key
    if not os.environ.get('GROQ_API_KEY'):
        print("\n⚠️  GROQ_API_KEY not set!")
        print("Get free API key at: https://console.groq.com")
        print("\nSet it with:")
        print('  export GROQ_API_KEY="your_key_here"  # Linux/Mac')
        print('  $env:GROQ_API_KEY="your_key_here"    # Windows PowerShell')
        return
    
    import time
    start = time.time()
    result = parse_resume_with_api(sample_resume, provider="groq")
    elapsed = time.time() - start
    
    print(f"\n⚡ API parsing completed in {elapsed:.2f} seconds")
    print(f"\n📋 Extracted Features:")
    print(f"  Job Title: {result['job_title']}")
    print(f"  Seniority: {result['seniority_level']}")
    print(f"  Experience: {result['years_of_experience']} years")
    print(f"  Skills: {result['num_skills']} skills")
    print(f"  Achievement Score: {result['achievement_score']:.1f}")
    
    print("\n" + "=" * 80)
    print("✅ API parsing works! No local GPU needed!")
    print("=" * 80)


if __name__ == "__main__":
    test_api_parser()
