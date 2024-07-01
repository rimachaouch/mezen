from flask import Flask, request, jsonify, send_from_directory
import PyPDF2
import pdfplumber
import spacy
import re
import os
import tempfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyresparser import ResumeParser
from werkzeug.utils import secure_filename
import nltk
import numpy as np
import joblib
import fitz  # PyMuPDF
from PIL import Image
import io
import uuid
from flask_cors import CORS, cross_origin
import smtplib

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Charger le modèle SVM
svm_model = joblib.load('best_calibrated_svm_model (1).pkl')
encoder = joblib.load("encoder (3).pkl")
imputer = joblib.load("imputer (2).pkl")

# Charger le modèle SpaCy pour le français
nlp = spacy.load("fr_core_news_sm")

UPLOAD_FOLDER = 'uploads/'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_IMAGES_FOLDER'] = EXTRACTED_IMAGES_FOLDER

# Vérifier si le fichier est autorisé
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction pour extraire les informations pertinentes d'un CV au format PDF
def extract_information_from_pdf(resume_path):
    try:
        parser = ResumeParser(resume_path)
        data = parser.get_extracted_data()

        education_info = extract_education(resume_path)
        languages = extract_languages(resume_path)

        with open(resume_path, 'rb') as pdf_file:
            images = extract_images_from_pdf(pdf_file.read())

        phone_number = extract_telephone_numbers(resume_path)

        cv_info = {
            'Email': data.get('email', 'N/A'),
            'Tel': phone_number,
            'Expérience': data.get('total_experience', 'N/A'),
            'Années_Expérience': data.get('total_experience', 'N/A'),
            'Diplôme': ', '.join(education_info),
            'Langues': ', '.join(languages),
            'Compétences': ', '.join(data.get('skills', [])),
            'Images': images
        }

        return cv_info
    except Exception as e:
        print("Une erreur s'est produite lors de l'extraction des informations du CV :", str(e))
        return None

# Fonction pour extraire les images d'un fichier PDF
def extract_images_from_pdf(pdf_stream):
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    images = []

    if not os.path.exists(app.config['EXTRACTED_IMAGES_FOLDER']):
        os.makedirs(app.config['EXTRACTED_IMAGES_FOLDER'])

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            # Générer un nom de fichier unique pour chaque image
            image_filename = f'image_{uuid.uuid4().hex}.{image_ext}'
            image_path = os.path.join(app.config['EXTRACTED_IMAGES_FOLDER'], image_filename)
            image.save(image_path)

            images.append(image_filename)

    return images

# Fonction pour extraire les numéros de téléphone à partir d'un PDF
def extract_telephone_numbers(pdf_path):
    tel_numbers_set = set()  # Utiliser un ensemble pour éviter les doublons
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            telephone_pattern = re.compile(r'\b\d{2} \d{3} \d{3}\b')
            telephone_matches = telephone_pattern.findall(text)
            tel_numbers_set.update(telephone_matches)  # Ajouter les numéros de téléphone à l'ensemble
    return list(tel_numbers_set) 

# Fonction pour extraire les informations sur l'éducation à partir d'un PDF
def extract_education(pdf_path):
    education_info = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            education_pattern = re.compile(r'(?:\bDiplôme\b|\bmaster\b|\blicence\b|\bBaccalauréat\b)[\s\w,.-]+?(?:\d{4})', re.IGNORECASE)

            #education_pattern = re.compile(r'(?:\bDiplôme\b|\bMaster\b|\blicence\b|\bBaccalauréat\b)[\s\w,.-]+', re.IGNORECASE)
            education_matches = education_pattern.findall(text)
            education_info.extend(education_matches)
    return education_info

# Fonction pour extraire les langues à partir d'un PDF
def extract_languages(pdf_path):
    languages = set()  # Utiliser un ensemble pour stocker les langues uniques
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            language_pattern = re.compile(r'(?:\bAnglais\b|\bFrançais\b|\bArabe\b|\bEspagnol\b|\bAllemand\b)', re.IGNORECASE)
            language_matches = language_pattern.findall(text)
            languages.update(language_matches)  # Mettre à jour l'ensemble avec les nouvelles langues trouvées
    return list(languages)  # Convertir l'ensemble en liste avant de le retourner


# Fonction pour calculer la similarité cosinus entre le texte du CV et la description de poste
def calculate_similarity(resume_text, job_description):
    resume_tokens = word_tokenize(resume_text, language='french')
    job_tokens = word_tokenize(job_description, language='french')

    stop_words = set(stopwords.words('french'))
    filtered_resume_tokens = [word for word in resume_tokens if word.lower() not in stop_words]
    filtered_job_tokens = [word for word in job_tokens if word.lower() not in stop_words]

    filtered_resume = ' '.join(filtered_resume_tokens)
    filtered_job = ' '.join(filtered_job_tokens)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered_resume, filtered_job])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]

# Fonction pour matcher le CV à la description de poste
def match_resume_to_job(resume_content, job_description, expert_threshold=80, medium_threshold=60):
    similarity_score = calculate_similarity(resume_content, job_description)
    similarity_percentage = similarity_score * 100
    
    if similarity_percentage >= expert_threshold:
        return "Débutant", similarity_percentage
    elif similarity_percentage >= medium_threshold:
        return "Medium", similarity_percentage
    else:
        return "Expert", similarity_percentage

# Fonction pour calculer le score de correspondance des compétences
def calculate_skill_match_score(cv_skills, job_skills):
    """
    Calcule le score de correspondance des compétences entre les compétences extraites du CV et celles requises pour le poste.

    Args:
        cv_skills (list): Liste des compétences extraites du CV.
        job_skills (list): Liste des compétences requises pour le poste.

    Returns:
        float: Le score de correspondance des compétences, arrondi à 2 décimales.
    """
    match_count = len(set(cv_skills) & set(job_skills))
    skill_match_score = (match_count / len(job_skills)) * 100
    return round(skill_match_score, 2)

# Fonction pour préparer les données d'entrée pour la prédiction
def prepare_input_text(cv_info, encoder):
    """
    Préparer les données d'entrée pour la prédiction en utilisant uniquement les caractéristiques catégorielles.

    Args:
        cv_info (dict): Dictionnaire contenant les informations du CV.
        encoder (LabelEncoder): L'encodeur pour les caractéristiques catégorielles.

    Returns:
        str: Une chaîne de caractères représentant les caractéristiques encodées.
    """
    categorical_features = ['Compétences', 'Langues', 'Diplôme']
    categorical_data = [cv_info[feature] for feature in categorical_features]
    categorical_data = np.array(categorical_data).reshape(1, -1)
    encoded_features = encoder.transform(categorical_data)
    input_text = ' '.join(map(str, encoded_features.ravel()))
    return input_text


# Fonction pour préparer les données d'entrée pour la prédiction
def prepare_input_textt(cv_info, encoder, imputer):
    categorical_features = ['Compétences', 'Langues', 'Diplôme']
    categorical_data = [cv_info[feature] for feature in categorical_features]
    categorical_data = np.array(categorical_data).reshape(1, -1)
    numerical_features = imputer.transform([[cv_info['Années_Expérience']]])
    encoded_features = encoder.transform(categorical_data)
    input_features = np.hstack((encoded_features, numerical_features))
    input_text = ' '.join(map(str, input_features.ravel()))
    return input_text

# Route pour télécharger la description de poste et les CV
@app.route('/compare', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def upload_files():
    if 'cvFiles' not in request.files or 'jobDescriptionFile' not in request.files:
        return jsonify({'error': 'Les fichiers sont manquants'})

    cv_files = request.files.getlist('cvFiles')
    job_description_file = request.files['jobDescriptionFile']

    if not cv_files or not job_description_file:
        return jsonify({'error': 'Nom de fichier vide'})

    if job_description_file and allowed_file(job_description_file.filename):
        job_description_filename = secure_filename(job_description_file.filename)
        job_description_file_path = os.path.join(app.config['UPLOAD_FOLDER'], job_description_filename)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        job_description_file.save(job_description_file_path)

        job_description_text = ""
        with pdfplumber.open(job_description_file_path) as pdf:
            for page in pdf.pages:
                job_description_text += page.extract_text()

        results = []
        for cv_file in cv_files:
            if cv_file and allowed_file(cv_file.filename):
                cv_filename = secure_filename(cv_file.filename)
                cv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
                cv_file.save(cv_file_path)

                cv_info = extract_information_from_pdf(cv_file_path)
                cv_content = ""
                with pdfplumber.open(cv_file_path) as pdf:
                    for page in pdf.pages:
                        cv_content += page.extract_text()

                result, score = match_resume_to_job(cv_content, job_description_text)
                score = round(score, 2)
                cv_skills = cv_info['Compétences'].split(', ')
                skill_match_score = calculate_skill_match_score(cv_skills, job_description_text)
                cv_info_dict = {
                    'Email': cv_info.get('Email', 'N/A'),
                    'Tel': cv_info.get('Tel', 'N/A'),
                    'Expérience': cv_info.get('Expérience', 'N/A'),
                    'Diplôme': cv_info.get('Diplôme', 'N/A'),
                    'Langues': cv_info.get('Langues', 'N/A'),
                    'Compétences': cv_info.get('Compétences', 'N/A'),
                    'Texte_CV': cv_info,
                    #'Images': [f"{request.host_url}{app.config['EXTRACTED_IMAGES_FOLDER']}/{image}" for image in cv_info['Images']],
                    'Description_Poste': job_description_text,
                    'Résultat': result,
                    'Score': score,
                    'File_Path': cv_file_path,
                    'skill_match_score':skill_match_score
                }

                results.append(cv_info_dict)

        return jsonify(results)
    else:
        return jsonify({'error': 'Format de fichier non autorisé pour l\'un ou les deux fichiers'})

# Route pour prédire la validité des CV
@app.route('/predict', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def upload_cv():
    if 'cvFiles' not in request.files:
        return jsonify({'error': 'Les fichiers sont manquants'})

    cv_files = request.files.getlist('cvFiles')

    if not cv_files:
        return jsonify({'error': 'Nom de fichier vide'})

    results = []
    for cv_file in cv_files:
        if cv_file and allowed_file(cv_file.filename):
            cv_filename = secure_filename(cv_file.filename)
            cv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
            cv_file.save(cv_file_path)

            cv_info = extract_information_from_pdf(cv_file_path)
            if cv_info:
                input_text = prepare_input_text(cv_info, encoder)
                input_features = list(map(float, input_text.split()))  # Convertir la chaîne en liste de nombres flottants
                input_features_array = np.array(input_features)
                input_features_2d = input_features_array.reshape(1, -1)  # Redimensionner le tableau en 2D
                predicted_status_proba = svm_model.predict_proba(input_features_2d)[0]  # Effectuer la prédiction de probabilité

                prediction = svm_model.predict(input_features_2d)[0]  # Effectuer la prédiction binaire
                max_proba = max(predicted_status_proba) * 100  # Obtenir la probabilité maximale et la convertir en pourcentage
                predict_statut = 'Validé' if max_proba >= 55 else 'Réfusé'
                
                cv_info_dict = {
                    'Email': cv_info.get('Email', 'N/A'),
                    'Tel': cv_info.get('Tel', 'N/A'),
                    'Expérience': cv_info.get('Expérience', 'N/A'),
                    'Diplôme': cv_info.get('Diplôme', 'N/A'),
                    'Langues': cv_info.get('Langues', 'N/A'),
                    'Compétences': cv_info.get('Compétences', 'N/A'),
                    'Texte_CV': cv_info,
                    #'Images': [f"{request.host_url}{app.config['EXTRACTED_IMAGES_FOLDER']}/{image}" for image in cv_info['Images']],
                    'predict_statut': predict_statut,
                    'predict_proba': round(max_proba, 2),  # Arrondir la probabilité maximale à deux décimales
                }
                results.append(cv_info_dict)
        else:
            return jsonify({'error': 'Format de fichier non autorisé pour l\'un ou les deux fichiers'})
    return jsonify(results)


# Route pour servir les images extraites
@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(app.config['EXTRACTED_IMAGES_FOLDER'], filename, as_attachment=False)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
