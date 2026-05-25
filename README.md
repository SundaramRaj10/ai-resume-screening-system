# AI-Based Resume Screening System

A professional resume screening dashboard built with Python, Streamlit, Pandas, Scikit-learn, TF-IDF vectorization, cosine similarity, and PDF text extraction.

## What It Does

- Accepts required job skills for a target role
- Screens a single candidate using pasted resume text or an uploaded PDF resume
- Matches candidate skills with required skills
- Calculates skill coverage and TF-IDF cosine similarity
- Produces a weighted final match score
- Predicts whether the candidate is suitable
- Shows matched skills, missing skills, contact details, and interview focus areas
- Downloads a candidate screening report
- Supports batch screening with CSV upload and result export

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- pypdf
- TF-IDF Vectorizer
- Cosine Similarity

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How The Score Works

The app combines two signals:

- 70% skill coverage: exact required skills found in the resume
- 30% TF-IDF similarity: text similarity between the required skills and resume

If the final score is greater than or equal to the selected threshold, the candidate is predicted as suitable.

## Batch CSV Format

Upload a CSV with these columns:

```csv
candidate_name,resume_text
Anika,"Python, Pandas, AI project experience"
Rahul,"Java, SQL, backend internship"
```

## Interview Explanation

This project uses NLP to compare a candidate resume against a job skill profile. Required skills are extracted from recruiter input, resume text is read from either text input or PDF, and TF-IDF converts both documents into numerical vectors. Cosine similarity measures how close the resume is to the job profile. A final weighted score combines exact skill coverage with semantic text similarity.
