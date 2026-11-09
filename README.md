# IdentityShield AI: Protect Your Digital Identity In The AI Era

IdentityShield AI is a professional, production-grade cybersecurity application that detects whether an uploaded image is authentic (Real) or AI-generated (Deepfake). It runs high-speed convolutional neural network classification (EfficientNetB0) in the backend to identify manipulated facial structures and artificial textures. 

The platform also includes a secondary web phishing scanner to detect malicious links and a reporting portal, providing a multi-layered shield to protect users from modern identity theft, digital scams, and social media misinformation.

---

## Key Features

- **AI Deepfake Image Detection**: Uses pre-trained deep learning weights mapping to an EfficientNetB0 backbone to identify synthetic media.
- **Phishing URL Scanner**: An integrated Logistic Regression + TF-IDF model that analyzes target URLs for identity scams and phishing hazards.
- **Glassmorphic SaaS Dashboard UI**: Sleek, modern, and dark-themed user interface styled with custom CSS gradients, responsive components, and fluid micro-animations (AOS).
- **Persistent Audit Logs**: Integrates with MongoDB to store client prediction outcomes, timestamps, and confidence percentages. Features a clean search, category filter, and paginated history table.
- **Fallback Logging Architecture**: Detects MongoDB availability on launch; if offline, automatically falls back to secure session/in-memory logs so the app remains fully functional.
- **Suspicious Media Reporting**: Allows citizens to file reports with scam details and images, logged directly to the incident database.
- **Fully Responsive**: Optimized for desktop, tablet, and mobile browsers.

---

## Directory Folder Structure

```text
IdentityShield/
│
├── app.py                  # Main Flask application with routing & model loader
├── config.py               # Settings (Upload limits, Allowed types, Paths)
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (DB connection, Keys)
├── README.md               # Application documentation
│
├── model/                  # Deep learning & machine learning classifiers
│   ├── DeepfakeWeights.h5  # Pre-trained CNN weights (EfficientNetB0)
│   ├── url_model.pkl       # Logistic regression URL model
│   └── vectorizer.pkl      # TF-IDF vectorizer for URLs
│
├── uploads/
│   └── images/             # Uploaded image files storage directory
│
├── static/
│   ├── css/
│   │   └── style.css       # Core stylesheet (Dark mode, glassmorphism)
│   ├── js/
│   │   └── script.js       # Client interactive events (AJAX, Drag-and-drop, filter)
│   └── images/
│       └── logo.png        # Icon representation placeholder
│
└── templates/
    ├── base.html           # Skeleton template (Navbar, Footer, script tags)
    ├── home.html           # Landing page with hero banner & workflow timeline
    ├── predict.html        # Main verification panel (Image upload & URL check)
    ├── result_real.html    # Success result view (Authentic outcomes)
    ├── result_fake.html    # Warning result view (Deepfake alert)
    ├── about.html          # Details regarding deepfakes and security
    ├── awareness.html      # Security portal cards mapping common scam tactics
    ├── history.html        # Log dashboard with search, filter, and page buttons
    ├── report.html         # Suspicious media abuse reporting forms
    ├── faq.html            # 10 comprehensive cyber security FAQs
    ├── contact.html        # Inquiries page with office coordinates
    └── 404.html            # Custom error redirect template
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 (highly recommended as TensorFlow does not support Python 3.14 yet)
- MongoDB installed locally or a MongoDB Atlas URI (optional; fallback mode will run automatically if offline)

### 2. Clone and Setup Environment

#### Windows PowerShell
```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install required libraries
venv\Scripts\python -m pip install -r requirements.txt
```

#### Linux / macOS Terminal
```bash
# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 3. Database Connection
Verify that MongoDB is running on `mongodb://localhost:27017` or configure your connection string in the `.env` file at the root directory:
```text
MONGO_URI=mongodb://localhost:27017/identityshield
SECRET_KEY=your_secret_production_key_here
PORT=5000
FLASK_ENV=development
```

---

## How To Run

1. Make sure your virtual environment is activated:
   - Windows: `venv\Scripts\activate`
   - Linux: `source venv/bin/activate`
2. Launch the Flask application server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```text
   http://localhost:5000
   ```

---

## Future Roadmap

- **Video Forgery Scanner**: Extract frame structures using OpenCV and pass sequence frames to recurrent neural networks (LSTM) to detect video deepfakes.
- **API Access**: Provide token-based endpoints for third-party developer integrations to check images programmatically.
- **Deepfake Audio Classifier**: Implement spectral voice analysis to detect synthetic clones used in phone call scams.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
