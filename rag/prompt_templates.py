
# rag/prompt_templates.py

# ============================================
# JOB ANALYZER PROMPTS
# ============================================

JOB_ANALYSIS_PROMPT = """
You are an expert HR analyst. Analyze this job description and extract:

1. Job title
2. Sector/Industry
3. Experience level required (junior/mid/senior)
4. Company context
5. Key responsibilities (list of 3-5 points)

Job Description:
{job_description}

Respond in JSON format only.
"""

# ============================================
# SKILL EXTRACTOR PROMPTS
# ============================================

SKILL_EXTRACTION_PROMPT = """
You are an expert technical recruiter. Extract all skills from this job description.

Categorize them as:
1. Hard skills - Must have (technical, mandatory)
2. Hard skills - Nice to have (technical, optional)
3. Soft skills (communication, leadership, etc.)

Job Description:
{job_description}

Respond in JSON format only:
{{
    "must_have": ["skill1", "skill2"],
    "nice_to_have": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"]
}}
"""

# ============================================
# PROFILE SEARCHER PROMPTS
# ============================================

BENCHMARK_PROFILE_PROMPT = """
You are an expert recruiter doing benchmarking.
Based on this job profile, describe what a successful candidate looks like.

Job Title: {job_title}
Required Skills: {must_have_skills}
Experience Level: {experience_level}

Provide a benchmark profile in JSON format only:
{{
    "ideal_background": "...",
    "typical_experience_years": "...",
    "key_past_roles": ["...", "..."],
    "differentiating_factors": ["...", "..."],
    "red_flags": ["...", "..."]
}}
"""

# ============================================
# CV GENERATOR PROMPTS
# ============================================

CV_GENERATION_PROMPT = """
You are an expert CV writer. Your goal is to generate an optimized CV 
for a candidate based on successful similar profiles.

JOB DESCRIPTION:
{job_description}

REQUIRED SKILLS:
{skills}

CANDIDATE PROFILE:
{candidate_profile}

SIMILAR SUCCESSFUL PROFILES (use as reference):
{context}

Generate a complete optimized CV with these sections:
1. Professional Summary (3-4 lines, tailored to the job)
2. Technical Skills (organized by category)
3. Professional Experience (highlight relevant achievements)
4. Education
5. Certifications (if relevant)

Make the CV ATS-friendly by naturally incorporating the required skills.
Respond in a clean, professional format.
"""