import re
from io import BytesIO
from typing import Iterable

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_THRESHOLD = 60
SKILL_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3
DEFAULT_SKILLS = "Python\nMachine Learning\nTensorFlow\nPandas\nData Preprocessing"
DEFAULT_RESUME = (
    "Built AI projects using Python and Pandas. Worked with data preprocessing, "
    "model evaluation, and basic machine learning. Created dashboards and cleaned datasets."
)


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon=":clipboard:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --app-bg: #eef3f8;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --text: #111827;
        --muted: #526173;
        --border: #cbd5e1;
        --primary: #0f766e;
        --primary-dark: #115e59;
        --accent: #1d4ed8;
    }
    .stApp {
        background: var(--app-bg);
        color: var(--text);
    }
    header[data-testid="stHeader"] {
        background: #0f172a;
    }
    .block-container {
        padding-top: 3.25rem;
        padding-bottom: 2.5rem;
        max-width: 1160px;
    }
    .main-title {
        color: var(--text);
        font-size: 2.05rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .subtle-text {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 1.25rem;
    }
    .section-label {
        color: #334155;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: var(--text);
    }
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 650 !important;
    }
    .stTextInput input,
    .stTextArea textarea {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel) !important;
        border: 1px dashed #94a3b8 !important;
        border-radius: 8px !important;
        min-height: 92px;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--panel-soft) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        color: var(--primary) !important;
        fill: none !important;
    }
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:focus:not(:active) {
        border-color: var(--primary-dark) !important;
        box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.2) !important;
        outline: none !important;
    }
    button[data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 650 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 2.75rem;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button[kind="primary"] *,
    div[data-testid="stButton"] button[kind="primary"] p {
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: var(--primary-dark) !important;
        border-color: var(--primary-dark) !important;
        color: #ffffff !important;
    }
    .fit-badge {
        border-radius: 8px;
        padding: 0.9rem 1rem;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        margin: 0.7rem 0 1rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .fit-title {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 750;
        margin: 0;
    }
    .fit-copy {
        color: var(--muted);
        margin: 0.2rem 0 0 0;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetric"] * {
        color: var(--text) !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 650 !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.7rem !important;
        font-weight: 750 !important;
        line-height: 1.2 !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: normal !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        color: #1e3a8a;
    }
    div[data-testid="stAlert"] * {
        color: inherit !important;
    }
    div[data-testid="stProgress"] > div {
        background-color: #dbeafe !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        background-color: var(--primary) !important;
    }
    .stSuccess {
        background: #ecfdf5 !important;
        color: #065f46 !important;
    }
    .stWarning {
        background: #fffbeb !important;
        color: #92400e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "resume_text_value" not in st.session_state:
    st.session_state.resume_text_value = DEFAULT_RESUME
if "last_pdf_name" not in st.session_state:
    st.session_state.last_pdf_name = None


def parse_skills(raw_skills: str) -> list[str]:
    """Convert comma, semicolon, or newline separated skills into a clean list."""
    skills = re.split(r"[\n,;]+", raw_skills)
    cleaned_skills = []

    for skill in skills:
        normalized = re.sub(r"\s+", " ", skill.strip().lower())
        if normalized and normalized not in cleaned_skills:
            cleaned_skills.append(normalized)

    return cleaned_skills


def extract_pdf_text(uploaded_pdf) -> str:
    reader = PdfReader(BytesIO(uploaded_pdf.getvalue()))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def skill_found(skill: str, resume_text: str) -> bool:
    normalized_resume = normalize_text(resume_text)
    normalized_skill = normalize_text(skill)

    if not normalized_skill:
        return False

    pattern = r"(?<![a-z0-9+#.])" + re.escape(normalized_skill) + r"(?![a-z0-9+#.])"
    return re.search(pattern, normalized_resume) is not None


def calculate_tfidf_score(required_skills: Iterable[str], resume_text: str) -> float:
    job_profile = " ".join(required_skills)

    if not job_profile.strip() or not resume_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([job_profile, resume_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


def get_fit_summary(final_score: float, threshold: int) -> tuple[str, str]:
    if final_score >= max(80, threshold):
        return "Strong Fit", "Candidate strongly matches the required skill profile."
    if final_score >= threshold:
        return "Good Fit", "Candidate meets the screening threshold and is worth shortlisting."
    if final_score >= threshold - 15:
        return "Needs Review", "Candidate is close to the threshold but has visible skill gaps."
    return "Not Recommended", "Candidate does not currently meet the required skill profile."


def screen_resume(required_skills: list[str], resume_text: str, threshold: int) -> dict:
    matched_skills = [skill for skill in required_skills if skill_found(skill, resume_text)]
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]

    skill_match_score = 0.0
    if required_skills:
        skill_match_score = round((len(matched_skills) / len(required_skills)) * 100, 2)

    tfidf_score = calculate_tfidf_score(required_skills, resume_text)
    final_score = round(
        (skill_match_score * SKILL_WEIGHT) + (tfidf_score * SIMILARITY_WEIGHT),
        2,
    )
    fit_label, fit_reason = get_fit_summary(final_score, threshold)

    return {
        "skill_match_score": skill_match_score,
        "tfidf_score": tfidf_score,
        "final_score": final_score,
        "prediction": "Suitable" if final_score >= threshold else "Not Suitable",
        "fit_label": fit_label,
        "fit_reason": fit_reason,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def build_candidate_report(candidate_name: str, job_title: str, company: str, result: dict) -> str:
    matched = ", ".join(skill.title() for skill in result["matched_skills"]) or "None"
    missing = ", ".join(skill.title() for skill in result["missing_skills"]) or "None"
    interview_focus = ", ".join(skill.title() for skill in result["missing_skills"][:3]) or "Validate project depth"

    return f"""AI Resume Screening Report

Candidate: {candidate_name or 'Candidate'}
Role: {job_title or 'Not specified'}
Company/Team: {company or 'Not specified'}

Final Match: {result['final_score']}%
Skill Coverage: {result['skill_match_score']}%
TF-IDF Similarity: {result['tfidf_score']}%
Prediction: {result['prediction']}
Fit Level: {result['fit_label']}

Matched Skills:
{matched}

Missing Skills:
{missing}

Recruiter Note:
{result['fit_reason']}

Suggested Interview Focus:
{interview_focus}
"""


def render_score_cards(result: dict) -> None:
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    top_left.metric("Final Match", f"{result['final_score']}%")
    top_right.metric("Prediction", result["prediction"])
    bottom_left.metric("Skill Coverage", f"{result['skill_match_score']}%")
    bottom_right.metric("TF-IDF", f"{result['tfidf_score']}%")

    st.progress(min(result["final_score"] / 100, 1.0))
    st.markdown(
        f"""
        <div class="fit-badge">
            <p class="fit-title">{result['fit_label']}</p>
            <p class="fit-copy">{result['fit_reason']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_lists(result: dict) -> None:
    matched_col, missing_col = st.columns(2)

    with matched_col:
        st.subheader("Matched Skills")
        if result["matched_skills"]:
            for skill in result["matched_skills"]:
                st.success(skill.title())
        else:
            st.info("No required skills found in the resume.")

    with missing_col:
        st.subheader("Missing Skills")
        if result["missing_skills"]:
            for skill in result["missing_skills"]:
                st.warning(skill.title())
        else:
            st.success("No missing required skills.")

def render_interview_focus(result: dict) -> None:
    st.subheader("Interview Focus")
    if result["missing_skills"]:
        focus_items = result["missing_skills"][:3]
        for skill in focus_items:
            st.write(f"- Validate practical knowledge of {skill.title()}.")
    else:
        st.write("- Discuss project depth, ownership, and real-world problem solving.")
        st.write("- Ask for examples of model evaluation and data preprocessing decisions.")


def build_batch_results(data: pd.DataFrame, required_skills: list[str], threshold: int) -> pd.DataFrame:
    rows = []

    for _, row in data.iterrows():
        result = screen_resume(required_skills, str(row["resume_text"]), threshold)
        rows.append(
            {
                "Candidate": row["candidate_name"],
                "Final Match (%)": result["final_score"],
                "Skill Coverage (%)": result["skill_match_score"],
                "TF-IDF Similarity (%)": result["tfidf_score"],
                "Prediction": result["prediction"],
                "Fit Level": result["fit_label"],
                "Matched Skills": ", ".join(result["matched_skills"]),
                "Missing Skills": ", ".join(result["missing_skills"]),
            }
        )

    return pd.DataFrame(rows).sort_values("Final Match (%)", ascending=False)


def render_batch_screening(required_skills: list[str], threshold: int) -> None:
    st.markdown('<p class="section-label">Batch Screening</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload candidate CSV",
        type=["csv"],
        help="Required columns: candidate_name, resume_text",
    )

    if uploaded_file is None:
        st.dataframe(pd.read_csv("sample_resumes.csv"), use_container_width=True, hide_index=True)
        return

    data = pd.read_csv(uploaded_file)
    required_columns = {"candidate_name", "resume_text"}

    if not required_columns.issubset(data.columns):
        st.error("CSV must contain candidate_name and resume_text columns.")
        return

    results = build_batch_results(data, required_skills, threshold)
    suitable_count = int((results["Prediction"] == "Suitable").sum())
    top_score = float(results["Final Match (%)"].max()) if not results.empty else 0.0

    batch_col_1, batch_col_2, batch_col_3 = st.columns(3)
    batch_col_1.metric("Candidates", len(results))
    batch_col_2.metric("Suitable", suitable_count)
    batch_col_3.metric("Top Match", f"{top_score}%")

    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Screening Results",
        data=results.to_csv(index=False),
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown('<div class="main-title">AI-Based Resume Screening System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Screen candidates with skill matching, TF-IDF similarity, and a clear suitability decision.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Settings")
    threshold = st.slider("Suitability threshold", 0, 100, DEFAULT_THRESHOLD, 5)
    st.caption("Final score = 70% skill coverage + 30% TF-IDF similarity")
    st.divider()
    st.header("Project Stack")
    st.write("Python")
    st.write("Streamlit")
    st.write("Scikit-learn")
    st.write("Pandas")
    st.write("pypdf")

job_col, company_col, candidate_col = st.columns(3)
with job_col:
    job_title = st.text_input("Job title", value="AI / ML Intern")
with company_col:
    company = st.text_input("Company or team", value="Data Science Team")
with candidate_col:
    candidate_name = st.text_input("Candidate name", value="Candidate")

required_skills_text = st.text_area(
    "Required job skills",
    value=DEFAULT_SKILLS,
    height=140,
)
required_skills = parse_skills(required_skills_text)

single_tab, batch_tab = st.tabs(["Single Candidate", "Batch Candidates"])

with single_tab:
    input_col, result_col = st.columns([1.05, 1.2])

    with input_col:
        uploaded_pdf = st.file_uploader("Upload resume PDF", type=["pdf"])

        if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.last_pdf_name:
            try:
                extracted_text = extract_pdf_text(uploaded_pdf)
                st.session_state.last_pdf_name = uploaded_pdf.name
                if extracted_text:
                    st.session_state.resume_text_value = extracted_text
                    st.success("PDF text extracted successfully.")
                else:
                    st.warning("No readable text found in this PDF. Paste the resume text below.")
            except Exception as error:
                st.error(f"Could not read this PDF: {error}")

        resume_text = st.text_area(
            "Candidate resume text",
            key="resume_text_value",
            height=330,
        )

        screen_clicked = st.button("Screen Candidate", type="primary", use_container_width=True)

    with result_col:
        if screen_clicked:
            if not required_skills:
                st.error("Enter at least one required skill.")
            elif not resume_text.strip():
                st.error("Enter resume text or upload a readable PDF resume.")
            else:
                st.session_state.screening_result = screen_resume(required_skills, resume_text, threshold)

        result = st.session_state.get("screening_result")
        if result:
            render_score_cards(result)
            render_skill_lists(result)
            render_interview_focus(result)
            report = build_candidate_report(candidate_name, job_title, company, result)
            st.download_button(
                "Download Candidate Report",
                data=report,
                file_name="candidate_screening_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("Run screening to generate the candidate report.")

with batch_tab:
    render_batch_screening(required_skills, threshold)




