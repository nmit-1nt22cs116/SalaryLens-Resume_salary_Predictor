"""
Groq API Salary Predictor
==========================
Uses Groq LLM to predict salary based on resume features
Provides explanations and insights alongside predictions

This is an OPTIONAL alternative to the XGBoost model
"""

import os
import json
from typing import Dict, Optional


def predict_salary_with_groq(form_data: dict, api_key: Optional[str] = None) -> dict:
    """
    Predict salary using Groq API
    
    Args:
        form_data: Resume features (same as XGBoost input)
        api_key: Optional Groq API key
    
    Returns:
        dict with prediction, explanation, and insights
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")
    
    api_key = api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    client = Groq(api_key=api_key)
    
    # Prepare resume summary for LLM
    resume_summary = f"""
Resume Profile:
- Job Title: {form_data.get('job_title')}
- Seniority: {form_data.get('seniority_level')}
- Experience: {form_data.get('years_of_experience')} years
- Education: {form_data.get('education_level')} in {form_data.get('field_of_study')}
- GPA: {form_data.get('gpa')}
- Location: {form_data.get('location')}
- Industry: {form_data.get('industry')}
- Company Size: {form_data.get('company_size')}

Skills: {form_data.get('skills')}
Number of Skills: {form_data.get('num_skills')}

Experience Metrics:
- Projects: {form_data.get('num_projects')}
- Publications: {form_data.get('num_publications')}
- Internships: {form_data.get('num_internships')}

Achievements:
- Leadership Experience: {'Yes' if form_data.get('has_leadership_experience') else 'No'}
- Open Source Contributions: {'Yes' if form_data.get('has_open_source_contributions') else 'No'}
- Certifications: {form_data.get('certifications')}
"""

    prompt = f"""You are an expert salary analyst. Based on this resume profile, predict the annual salary in USD.

{resume_summary}

Provide your analysis as JSON:
{{
  "predicted_salary": 120000,
  "salary_range": {{"low": 100000, "high": 140000}},
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of salary factors",
  "key_factors": ["factor1", "factor2", "factor3"],
  "recommendations": "How to increase salary",
  "market_insights": "Current market trends for this role"
}}

Consider:
1. Industry standards for this role and seniority
2. Location cost of living and market rates
3. Skills value (especially tech skills like Python, AWS, etc.)
4. Experience level and achievements
5. Education and certifications
6. Company size and industry

Be realistic and data-driven. Return ONLY the JSON.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Updated model (replaces decommissioned llama-3.1-70b-versatile)
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Lower for more consistent predictions
        max_tokens=800,
    )
    
    response_text = response.choices[0].message.content
    
    # Extract JSON
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        result = json.loads(json_str)
        
        # Ensure all required fields
        result.setdefault('predicted_salary', 100000)
        result.setdefault('salary_range', {'low': 90000, 'high': 110000})
        result.setdefault('confidence', 'medium')
        result.setdefault('reasoning', 'Based on market analysis')
        result.setdefault('key_factors', [])
        result.setdefault('recommendations', 'Continue building skills')
        result.setdefault('market_insights', 'Market is competitive')
        
        return result
    else:
        raise ValueError("No valid JSON in response")


def compare_predictions(xgboost_result: dict, groq_result: dict) -> dict:
    """
    Compare XGBoost and Groq predictions
    
    Returns:
        dict with comparison analysis
    """
    xgb_pred = xgboost_result['predicted']
    groq_pred = groq_result['predicted_salary']
    
    difference = abs(xgb_pred - groq_pred)
    percent_diff = (difference / xgb_pred) * 100
    
    # Determine agreement level
    if percent_diff < 5:
        agreement = "Strong Agreement"
        confidence = "Very High"
    elif percent_diff < 10:
        agreement = "Good Agreement"
        confidence = "High"
    elif percent_diff < 20:
        agreement = "Moderate Agreement"
        confidence = "Medium"
    else:
        agreement = "Significant Difference"
        confidence = "Low"
    
    return {
        'xgboost_prediction': xgb_pred,
        'groq_prediction': groq_pred,
        'difference': difference,
        'percent_difference': percent_diff,
        'agreement': agreement,
        'confidence': confidence,
        'average_prediction': (xgb_pred + groq_pred) / 2,
        'recommendation': get_recommendation(percent_diff, xgb_pred, groq_pred)
    }


def get_recommendation(percent_diff: float, xgb: float, groq: float) -> str:
    """Get recommendation based on prediction difference"""
    
    if percent_diff < 5:
        return "Both models agree strongly. This prediction is highly reliable."
    elif percent_diff < 10:
        return "Models show good agreement. Consider the average as a reliable estimate."
    elif percent_diff < 20:
        return "Models show moderate difference. The true salary likely falls between the two predictions."
    else:
        higher = "XGBoost" if xgb > groq else "Groq"
        lower = "Groq" if xgb > groq else "XGBoost"
        return f"Significant difference detected. {higher} predicts higher. Consider market research for this specific role."


def test_groq_predictor():
    """Test the Groq salary predictor"""
    
    print("=" * 80)
    print("TESTING GROQ SALARY PREDICTOR")
    print("=" * 80)
    
    # Sample resume data
    sample_data = {
        'job_title': 'Senior Machine Learning Engineer',
        'seniority_level': 'Senior',
        'years_of_experience': 8,
        'education_level': 'Master',
        'field_of_study': 'Computer Science',
        'gpa': 3.8,
        'skills': 'Python|TensorFlow|PyTorch|AWS|Docker|Kubernetes',
        'num_skills': 6,
        'location': 'San Francisco, CA',
        'industry': 'Technology',
        'company_size': 'Enterprise (5000+)',
        'num_projects': 8,
        'num_publications': 3,
        'num_internships': 1,
        'has_leadership_experience': 1,
        'has_open_source_contributions': 1,
        'certifications': 'AWS Certified'
    }
    
    if not os.environ.get('GROQ_API_KEY'):
        print("\n❌ GROQ_API_KEY not set")
        print("Set it with: python setup_api_key.py")
        return
    
    print("\nPredicting salary with Groq API...")
    
    try:
        result = predict_salary_with_groq(sample_data)
        
        print("\n" + "=" * 80)
        print("GROQ PREDICTION RESULTS")
        print("=" * 80)
        
        print(f"\n💰 Predicted Salary: ${result['predicted_salary']:,}")
        print(f"📊 Salary Range: ${result['salary_range']['low']:,} - ${result['salary_range']['high']:,}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        print(f"\n📝 Reasoning:")
        print(f"   {result['reasoning']}")
        
        print(f"\n🔑 Key Factors:")
        for factor in result['key_factors']:
            print(f"   • {factor}")
        
        print(f"\n💡 Recommendations:")
        print(f"   {result['recommendations']}")
        
        print(f"\n📈 Market Insights:")
        print(f"   {result['market_insights']}")
        
        print("\n" + "=" * 80)
        print("✅ GROQ PREDICTOR WORKS!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_groq_predictor()
