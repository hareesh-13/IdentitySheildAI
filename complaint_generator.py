import os
import datetime
import cv2
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib.units import inch

def extract_video_frame(video_path, output_image_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_image_path, frame)
        cap.release()
        return ret
    except Exception as e:
        print(f"Error extracting video frame: {e}")
        return False

def generate_audio_waveform(audio_path, output_image_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        plt.figure(figsize=(6, 2))
        plt.plot(y, color="#7c3aed", linewidth=0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating waveform: {e}")
        return False

def generate_complaint_pdf(report_data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterHeading', alignment=1, fontSize=16, spaceAfter=20, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='SectionTitle', fontSize=12, spaceAfter=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#0284c7")))
    styles.add(ParagraphStyle(name='NormalText', fontSize=10, spaceAfter=12, leading=14))
    
    elements = []
    
    # Header
    elements.append(Paragraph("IdentityShield AI - Official Complaint Copy", styles['CenterHeading']))
    elements.append(Paragraph("AI Detection Verification & Incident Report", styles['CenterHeading']))
    elements.append(Spacer(1, 20))
    
    # 1. Complaint Information
    elements.append(Paragraph("1. Complaint Information", styles['SectionTitle']))
    data_info = [
        ["Complaint ID:", report_data.get('complaint_id', 'N/A')],
        ["Date/Time Submitted:", report_data.get('timestamp', 'N/A')],
        ["Incident Category:", report_data.get('incident_type', 'N/A')],
        ["Current Status:", report_data.get('status', 'N/A')],
    ]
    t_info = Table(data_info, colWidths=[150, 300])
    t_info.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 15))
    
    # 2. Complainant Information
    elements.append(Paragraph("2. Complainant Information", styles['SectionTitle']))
    data_user = [
        ["Name:", report_data.get('name', 'N/A')],
        ["Contact Email:", report_data.get('email', 'N/A')],
        ["Mobile Number:", report_data.get('mobile', 'N/A')],
        ["Residential Address:", report_data.get('address', 'N/A')],
    ]
    t_user = Table(data_user, colWidths=[150, 300])
    t_user.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_user)
    elements.append(Spacer(1, 15))
    
    # 3. Incident Description
    elements.append(Paragraph("3. Incident Details", styles['SectionTitle']))
    platform_txt = report_data.get('platform', 'N/A')
    elements.append(Paragraph(f"<b>Platform/URL:</b> {platform_txt}", styles['NormalText']))
    desc_txt = report_data.get('description', 'N/A')
    elements.append(Paragraph(f"<b>Description:</b> {desc_txt}", styles['NormalText']))
    elements.append(Spacer(1, 15))
    
    # 4. AI Detection Evidence
    elements.append(Paragraph("4. AI Detection Evidence (Verified)", styles['SectionTitle']))
    data_ai = [
        ["Detection System:", f"IdentityShield AI - {report_data.get('detection_type', 'Manual Upload')}"],
        ["Filename:", report_data.get('filename', 'N/A')],
        ["Prediction Result:", "Deepfake (Fake)" if report_data.get('score') else "N/A"],
        ["Confidence Score:", f"{report_data.get('confidence', 'N/A')}%"],
        ["Analysis Timestamp:", report_data.get('detection_timestamp', 'N/A')],
    ]
    t_ai = Table(data_ai, colWidths=[150, 300])
    t_ai.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f9ff")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd"))
    ]))
    elements.append(t_ai)
    elements.append(Spacer(1, 15))
    
    confidence_str = report_data.get('confidence', '')
    score_str = report_data.get('score', '')
    det_type = report_data.get('detection_type', '')
    
    if score_str and confidence_str:
        try:
            score_val = float(score_str)
            conf_val = float(confidence_str)
        except:
            score_val = 0.0
            conf_val = 0.0
            
        if conf_val >= 80:
            risk = "High"
            tier = "High"
        elif conf_val >= 60:
            risk = "High"
            tier = "Medium"
        else:
            risk = "Medium"
            tier = "Low"
            
        elements.append(Paragraph("Prediction Summary Replica", styles['SectionTitle']))
        
        box_style = ParagraphStyle(name='BoxStyle', alignment=1, fontSize=12, fontName="Helvetica-Bold")
        sub_style = ParagraphStyle(name='SubStyle', alignment=1, fontSize=8, fontName="Helvetica", textColor=colors.gray)
        
        p_conf = [Paragraph(f"<font color='#dc2626'>{confidence_str}%</font>", box_style), Paragraph("CONFIDENCE", sub_style)]
        p_res = [Paragraph("<font color='#dc2626'>Fake</font>", box_style), Paragraph("RESULT", sub_style)]
        
        risk_hex = "#dc2626" if risk == "High" else "#f59e0b"
        p_risk = [Paragraph(f"<font color='{risk_hex}'>{risk}</font>", box_style), Paragraph("RISK", sub_style)]
        p_tier = [Paragraph(f"<font color='black'>{tier}</font>", box_style), Paragraph("TIER", sub_style)]
        
        summary_data = [[p_conf, p_res, p_risk, p_tier]]
        
        t_summary = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t_summary.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('TOPPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
        ]))
        
        elements.append(t_summary)
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Authenticity Score", styles['NormalText']))
        d = Drawing(400, 20)
        d.add(Rect(0, 5, 400, 10, fillColor=colors.HexColor("#f1f5f9"), strokeColor=colors.transparent))
        bar_width = 400 * score_val
        d.add(Rect(0, 5, bar_width, 10, fillColor=colors.HexColor("#dc2626"), strokeColor=colors.transparent))
        elements.append(d)
        
        score_text = f"0.0 (Fake) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Score: {score_str} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.0 (Real)"
        elements.append(Paragraph(f"<font size=8 color='gray'>{score_text}</font>", styles['NormalText']))
        elements.append(Spacer(1, 15))

    # 5. Media Evidence Attachment
    filename = report_data.get('filename')
    if filename:
        elements.append(Paragraph("5. Media Evidence Attachment", styles['SectionTitle']))
        ext = filename.split('.')[-1].lower()
        upload_folder = os.path.join(os.getcwd(), 'uploads', 'images')
        file_path = os.path.join(upload_folder, filename)
        
        if ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp'] and os.path.exists(file_path):
            try:
                img = Image(file_path)
                img._restrictSize(400, 300)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<para align='center'><i>Attached Image: {filename}</i></para>", styles['NormalText']))
            except Exception as e:
                elements.append(Paragraph(f"<i>[Image preview unavailable. Filename: {filename}]</i>", styles['NormalText']))
        elif ext in ['mp4', 'avi', 'mov', 'webm'] and os.path.exists(file_path):
            try:
                frame_path = os.path.join(upload_folder, f"preview_{filename}.jpg")
                if extract_video_frame(file_path, frame_path):
                    img = Image(frame_path)
                    img._restrictSize(400, 300)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"<para align='center'><i>Attached Video Frame: {filename}</i></para>", styles['NormalText']))
                else:
                    elements.append(Paragraph(f"<i>[Video preview unavailable. Filename: {filename}]</i>", styles['NormalText']))
            except Exception as e:
                pass
        elif ext in ['wav', 'mp3', 'flac', 'ogg', 'm4a'] and os.path.exists(file_path):
            try:
                wave_path = os.path.join(upload_folder, f"wave_{filename}.png")
                if generate_audio_waveform(file_path, wave_path):
                    img = Image(wave_path)
                    img._restrictSize(400, 150)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"<para align='center'><i>Attached Audio Waveform: {filename}</i></para>", styles['NormalText']))
                else:
                    elements.append(Paragraph(f"<i>[Audio preview unavailable. Filename: {filename}]</i>", styles['NormalText']))
            except Exception as e:
                pass
        else:
            elements.append(Paragraph(f"<b>Attached Media File:</b> {filename} (Preview not natively supported for {ext.upper()} format in this document. File is securely stored in evidence database.)", styles['NormalText']))
            
        elements.append(Spacer(1, 15))
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<i>This document is an auto-generated complaint copy verified by the IdentityShield AI platform. It serves as digital evidence of the detected synthetic media. Note: IdentityShield AI is a software platform and not a law enforcement authority.</i>", styles['NormalText']))
    
    doc.build(elements)
