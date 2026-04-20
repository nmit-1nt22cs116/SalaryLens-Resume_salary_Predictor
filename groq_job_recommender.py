"""
Groq Job Recommender for Streamlit App
=======================================
Intelligent job search recommendations using Groq AI

Usage:
    from groq_job_recommender import get_job_recommendations, enhance_job_search_query
"""

import os
import json
from typing import Dict, List, Optional

def get_job_recommendations(form_data: dict) -> dict:
    """
    Use Groq AI to analyze resume data and generate intelligent job search recommendations
    
    Args:
        form_data: Resume data from Streamlit form
    
    Returns:
        dict with recommendations including job titles, search queries, career advice
    """
    if not os.environ.get('GROQ_API_KEY'):
        raise ValueError("GROQ_API_KEY not set. Get free key at https://console.groq.com")
    
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")
    
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    
    # Build comprehensive profile for Groq
    skills = form_data.get('skills', '')
    if isinstance(skills, str):
        skills = skills.replace('|', ', ')
    
    profile = f"""
CANDIDATE PROFILE:
- Current Title: {form_data.get('job_title', 'Software Engineer')}
- Seniority: {form_data.get('seniority_level', 'Mid-level')}
- Experience: {form_data.get('years_of_experience', 3)} years
- Education: {form_data.get('education_level', 'Bachelor')} in {form_data.get('field_of_study', 'Computer Science')}
- Skills: {skills}
- Location: {form_data.get('location', 'San Francisco, CA')}
- Industry: {form_data.get('industry', 'Technology')}
- Leadership: {'Yes' if form_data.get('has_leadership_experience') else 'No'}
- Open Source: {'Yes' if form_data.get('has_open_source_contributions') else 'No'}
"""

    prompt = f"""You are an expert career advisor and job market analyst. Analyze this candidate profile and generate smart job search recommendations.

{profile}

Return ONLY valid JSON with no extra text:
{{
  "recommended_titles": ["title1", "title2", "title3", "title4", "title5"],
  "search_queries": [
    {{"query": "Senior Python Engineer", "reason": "Core skill match", "priority": "high"}},
    {{"query": "ML Engineer AWS", "reason": "Cloud + ML combo", "priority": "medium"}},
    {{"query": "Backend Engineer", "reason": "Experience level", "priority": "high"}}
  ],
  "top_skills_to_highlight": ["skill1", "skill2", "skill3"],
  "career_level_advice": "Brief advice on targeting the right level",
  "industries_to_target": ["Technology", "Finance", "Healthcare"],
  "salary_expectation": "Expected range based on profile",
  "profile_strengths": ["strength1", "strength2", "strength3"],
  "skill_gaps": ["gap1", "gap2"],
  "market_insights": "Current job market trends for this profile"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        
        text = response.choices[0].message.content
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start == -1:
            raise ValueError("No JSON found in Groq response")
        
        return json.loads(text[json_start:json_end])
    
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")


def enhance_job_search_query(form_data: dict, base_query: str) -> str:
    """
    Use Groq to enhance a basic job search query with intelligent keywords
    
    Args:
        form_data: Resume data
        base_query: Basic search query (e.g., "Software Engineer")
    
    Returns:
        Enhanced search query optimized for better results
    """
    if not os.environ.get('GROQ_API_KEY'):
        return base_query  # Fallback to original query
    
    try:
        from groq import Groq
        client = Groq(api_key=os.environ['GROQ_API_KEY'])
        
        skills = form_data.get('skills', '')
        if isinstance(skills, str):
            skills = skills.replace('|', ', ')
        seniority = form_data.get('seniority_level', 'Mid-level')
        experience = form_data.get('years_of_experience', 0)
        
        prompt = f"""Enhance this job search query for better results:

Base query: "{base_query}"
Candidate skills: {skills}
Seniority: {seniority}
Years of experience: {experience}

Return an enhanced search query (max 10 words) that includes the most relevant keywords for job search APIs. Focus on:
1. Seniority level
2. Years of experience (e.g., "5+ years")
3. Top 2-3 most marketable skills
4. Job title variations

Return ONLY the enhanced query, no explanation:"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        
        enhanced = response.choices[0].message.content.strip().strip('"')
        return enhanced if len(enhanced) > 0 else base_query
    
    except:
        return base_query  # Fallback on any error


def analyze_job_match(job_description: str, form_data: dict) -> dict:
    """
    Use Groq to analyze how well a job matches the candidate's profile
    
    Args:
        job_description: Job posting description
        form_data: Candidate's resume data
    
    Returns:
        dict with match analysis
    """
    if not os.environ.get('GROQ_API_KEY'):
        return {"match_score": 50, "match_reasons": ["Basic keyword match"], "missing_skills": []}
    
    try:
        from groq import Groq
        client = Groq(api_key=os.environ['GROQ_API_KEY'])
        
        candidate_skills = form_data.get('skills', '')
        if isinstance(candidate_skills, str):
            candidate_skills = candidate_skills.replace('|', ', ')
        experience = form_data.get('years_of_experience', 0)
        seniority = form_data.get('seniority_level', 'Mid-level')
        
        prompt = f"""Analyze job match considering experience requirements:

CANDIDATE:
- Skills: {candidate_skills}
- Experience: {experience} years
- Level: {seniority}

JOB DESCRIPTION:
{job_description[:800]}

Analyze if the candidate's experience level matches the job requirements. Consider:
1. Does the job specify years of experience required?
2. Is the candidate underqualified, qualified, or overqualified?
3. How well do skills align?

Return JSON:
{{
  "match_score": 85,
  "match_reasons": ["Strong Python skills", "Experience matches 5+ years requirement", "AWS experience"],
  "missing_skills": ["Kubernetes", "Go"],
  "experience_fit": "Good match - meets 5+ years requirement",
  "recommendation": "Apply - good fit"
}}"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        
        text = response.choices[0].message.content
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start != -1:
            return json.loads(text[json_start:json_end])
    
    except:
        pass
    
    # Fallback analysis
    return {
        "match_score": 50,
        "match_reasons": ["Basic compatibility"],
        "missing_skills": [],
        "recommendation": "Review job details"
    }
