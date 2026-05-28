from flask import Flask, render_template, request
import os
import re

from docx import Document
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# CREATE RESUME FOLDER
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'resumes')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SKILLS DATABASE
skills_list = [
    'python',
    'java',
    'machine learning',
    'sql',
    'html',
    'css',
    'flask',
    'django',
    'javascript',
    'data science',
    'c++',
    'react',
    'nodejs',
    'mongodb',
    'deep learning',
    'power bi',
    'excel'
]


# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')


# UPLOAD & ANALYZE
@app.route('/upload', methods=['POST'])
def upload_file():

    # GET MULTIPLE FILES
    files = request.files.getlist('resume')

    if not files:
        return "No files uploaded"

    # JOB DESCRIPTION
    job_description = request.form['job_description'].lower()

    results = []

    # PROCESS EACH RESUME
    for file in files:

        if file.filename == '':
            continue

        # SAVE FILE
        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )

        file.save(filepath)

        resume_text = ""

        # DOCX SUPPORT
        if file.filename.endswith('.docx'):

            doc = Document(filepath)

            for para in doc.paragraphs:
                resume_text += para.text + "\n"

        # PDF SUPPORT
        elif file.filename.endswith('.pdf'):

            with open(filepath, 'rb') as pdf_file:

                pdf_reader = PyPDF2.PdfReader(pdf_file)

                for page in pdf_reader.pages:

                    text = page.extract_text()

                    if text:
                        resume_text += text

        else:
            continue

        # LOWERCASE
        resume_text_lower = resume_text.lower()

        # -----------------------------
        # EXTRACT EMAIL
        # -----------------------------
        email_match = re.findall(
            r'[\w\.-]+@[\w\.-]+',
            resume_text
        )

        email = email_match[0] if email_match else "Not Found"

        # -----------------------------
        # EXTRACT PHONE NUMBER
        # -----------------------------
        phone_match = re.findall(
            r'\b\d{10}\b',
            resume_text
        )

        phone = phone_match[0] if phone_match else "Not Found"

        # -----------------------------
        # EXTRACT CANDIDATE NAME
        # -----------------------------
        lines = resume_text.split('\n')

        candidate_name = "Unknown"

        for line in lines:

            clean_line = line.strip()

            if len(clean_line) > 2 and len(clean_line) < 40:

                candidate_name = clean_line
                break

        # -----------------------------
        # DETECTED SKILLS
        # -----------------------------
        detected_skills = []

        for skill in skills_list:

            if skill in resume_text_lower:
                detected_skills.append(skill)

        # -----------------------------
        # MATCHED SKILLS
        # -----------------------------
        matched_skills = []

        for skill in skills_list:

            if (
                skill in job_description and
                skill in detected_skills
            ):
                matched_skills.append(skill)

        # -----------------------------
        # MISSING SKILLS
        # -----------------------------
        missing_skills = []

        for skill in skills_list:

            if (
                skill in job_description and
                skill not in detected_skills
            ):
                missing_skills.append(skill)

        # -----------------------------
        # NLP MATCHING
        # -----------------------------
        documents = [
            job_description,
            resume_text_lower
        ]

        tfidf = TfidfVectorizer()

        tfidf_matrix = tfidf.fit_transform(documents)

        similarity_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        match_percentage = similarity_score[0][0] * 100

        # -----------------------------
        # RECOMMENDATION ENGINE
        # -----------------------------
        if match_percentage >= 80:

            recommendation = "Excellent Candidate"

        elif match_percentage >= 60:

            recommendation = "Good Candidate"

        elif match_percentage >= 40:

            recommendation = "Average Candidate"

        else:

            recommendation = "Needs Improvement"

        # -----------------------------
        # SAVE RESULT
        # -----------------------------
        results.append({

            'name': candidate_name,

            'email': email,

            'phone': phone,

            'filename': file.filename,

            'score': round(match_percentage, 2),

            'matched_skills': matched_skills,

            'missing_skills': missing_skills,

            'recommendation': recommendation

        })

    # SORT RESULTS
    results = sorted(
        results,
        key=lambda x: x['score'],
        reverse=True
    )

    return render_template(
        'result.html',
        results=results
    )


# RUN APP
if __name__ == '__main__':
    app.run(debug=True)