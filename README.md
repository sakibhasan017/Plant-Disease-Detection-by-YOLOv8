# Plant Disease Detection 🌿

**Plant Disease Detection** is a web application built with **Streamlit** and **YOLOv8** to detect plant leaf diseases from uploaded images. The app identifies multiple leaves in an image, detects the disease, and provides **cause** and **cure/treatment suggestions**.

---

## Features

- Detect multiple leaves in a single image.  
- Identify plant diseases using a trained **YOLOv8** model.  
- Display **cause** and **cure/treatment** for each disease.  
- Automatically handles **healthy leaves**.  
- Modern, attractive **UI design** for better user experience.  
- Works entirely in a browser – no local setup required beyond dependencies.

---

## Demo

You can deploy your own version or run locally:

![Plant Doctor AI Screenshot](screenshot.png)  <!-- Optional screenshot -->

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/plant-disease-app.git
cd plant-disease-app
```

### 2. Create a Python virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the app locally
```bash
streamlit run web_app.py

```
The app will open in your default browser at http://localhost:8501.

## Deployment

The easiest way to deploy and share your app is using **Streamlit Community Cloud**:

1. Push your repository to **GitHub**.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/).  
3. Click **“New app”**, select your GitHub repo, branch, and `web_app.py`.  
4. Click **Deploy**.  
5. Share your **public URL** with anyone.

> ✅ Works on desktop, tablet, and mobile.

---

## Requirements

- Python 3.8+  
- Streamlit  
- Ultralytics YOLO  
- Pillow  
- NumPy  
- Roboflow (if using dataset API)

Example `requirements.txt`:

```text
streamlit
ultralytics
numpy
Pillow
roboflow
```

## Usage

1. Open the app in a browser.  
2. Upload an image of your plant leaves.  
3. Wait for the model to detect leaves.  
4. See the detection results, disease names, cause, and cure.  
5. Healthy leaves will display **“No treatment required.”**

---

## Notes

- Ensure `best.pt` (YOLOv8 model) is present in the project directory.  
- For large images, inference may take a few seconds.  
- The app supports multiple leaf detections and prints each disease **once**.  
