# AuraShop Support Bot & Visual Defect Detection Portal

A Full-Stack prototype of a modern Customer Support Chatbot integrated with a custom computer vision processing and defect classification pipeline. Built with **Python (FastAPI)** on the backend and a clean, responsive layout using **HTML, Tailwind CSS, and JavaScript** on the frontend.

## 🚀 Core Features
1. **Multi-Turn Chatbot**: Handles general e-commerce inquiries, return policies, order tracking, and guides users through return/exchange filings.
2. **Advanced CV Inspection Engine**: Uses K-means color clustering, connected component shape modeling (aspect ratios, convex hulls, shoelace area), and interface dilation tests to detect local defect structures (e.g., chipped parts, peeling gaps).
3. **Real-time Pipeline Visualizer**: Renders downscaled tensor shapes, image properties (contrast, edge density, brightness), and prints raw execution logs on a monitoring dashboard.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+ installed and added to your system PATH.
- Git (optional, for version control).

### Run Locally (Windows)
1. Double-click or execute the startup PowerShell script to create the virtual environment, install dependencies, and launch the server:
   ```powershell
   .\run.ps1
   ```
2. Open your web browser and navigate to:
   👉 **https://e-commerce-chatbot-nfrt.onrender.com**

---

## 📂 Project Structure
- `/app`
  - `main.py`: FastAPI server routing and multipart upload streams.
  - `bot.py`: Dialogue manager session state machine.
  - `vision.py`: Image preprocessors, segmenter, and geometric classifiers.
- `/static`
  - `index.html`: Main dashboard web page.
  - `app.js`: Script coordinating network requests, canvas rendering, and pipeline logs.
  - `style.css`: Clean layouts and styling classes.
- `/tests`
  - `test_backend.py`: Verification tests for backend logic.
- `.gitignore`: Excludes Python cache, environment directories, and image uploads.
- `requirements.txt`: Project dependencies list.
