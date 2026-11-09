import os
import uuid
import datetime
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from flask import current_app

def get_gemini_explanation(result, confidence, modality):
    api_key = current_app.config.get('GEMINI_API_KEY')
    if not api_key:
        return "AI Explanation unavailable. Please configure the Gemini API key."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"You are a cybersecurity and deepfake detection expert. A user uploaded a {modality} which our AI system analyzed. The result is {result} with a confidence score of {confidence}%. Provide a brief, professional explanation (max 3 sentences) of what this means and what the user should do next."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"This {modality} was determined to be {result} with {confidence}% confidence based on AI artifacts and analysis."

def generate_evidence_pdf(filename, modality, result, confidence, explanation):
    report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pdf_filename = f"{report_id}.pdf"
    pdf_path = os.path.join(current_app.config['PDF_REPORTS_FOLDER'], pdf_filename)
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "IdentityShield AI")
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 75, "AI Evidence & Detection Report")
    
    # Line
    c.setStrokeColor(colors.grey)
    c.line(50, height - 90, width - 50, height - 90)
    
    # Content
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 120, "Report Details:")
    
    c.setFont("Helvetica", 11)
    c.drawString(70, height - 140, f"Report ID: {report_id}")
    c.drawString(70, height - 160, f"Date & Time: {timestamp}")
    c.drawString(70, height - 180, f"Media Type: {modality.capitalize()}")
    c.drawString(70, height - 200, f"Source File: {filename}")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 230, "Detection Result:")
    
    if result.lower() == "fake":
        c.setFillColor(colors.red)
    else:
        c.setFillColor(colors.green)
        
    c.drawString(70, height - 250, f"Verdict: {result.upper()}")
    c.setFillColor(colors.black)
    c.drawString(70, height - 270, f"Confidence Score: {confidence}%")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 300, "AI Analysis & Explanation:")
    
    # Wrap text for explanation
    from reportlab.lib.utils import simpleSplit
    c.setFont("Helvetica", 11)
    lines = simpleSplit(explanation, "Helvetica", 11, width - 120)
    y_pos = height - 320
    for line in lines:
        c.drawString(70, y_pos, line)
        y_pos -= 15
        
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, 50, "This is an automatically generated AI analysis report. It does not constitute legal advice.")
    
    c.save()
    
    return report_id, pdf_filename
