# Turkish License Plate Recognition System

An application that detects a license plate in a vehicle photo and reads the text on it. Plate detection uses a YOLOv8 model that I trained myself, and the plate text is read with EasyOCR. The project was developed on Google Colab and is deployed as a Streamlit web app.

## Live Demo
👉 https://license-plate-recognition-1.streamlit.app/

The free-tier app goes to sleep after a period of inactivity. If you see a sleep screen, click "Yes, get this app back up!" and it wakes up in 1-2 minutes. The first photo may take a little longer because the models are loaded on first use.

## How It Works
1. **Data preparation:** A labeled license plate dataset in YOLO format is loaded from Google Drive. All labels are converted to a single class ("plate").
2. **Model training:** A pretrained YOLOv8n model is trained on this dataset for plate detection. The best model is saved as `best_plaka_model.pt`, so training does not need to be repeated.
3. **Detection:** The trained model finds the plate in the photo and draws a bounding box around it.
4. **Reading:** The plate region is cropped. The blue "TR" strip on the left and the edges are removed, the image is enlarged, and it is processed in three ways (grayscale, contrast-enhanced, black and white). Each version is read with EasyOCR.
5. **Validation and correction:** The text is checked against the Turkish plate format (2 digits + 1-3 letters + 2-4 digits). Characters that OCR often confuses are corrected by position, for example `O` to `0` in the province code or `0` to `O` in the letter part. The most confident result that matches the format is shown. Results that do not match the format are reported as unreadable.

## Project Files
| File | Description |
|---|---|
| `PlakaTahmin_1.ipynb` | Colab notebook: data preparation, model training, OCR experiments and a Gradio interface |
| `streamlit_app.py` | Streamlit web app used for the live demo |
| `best_plaka_model.pt` | Trained YOLOv8 plate detection model |
| `requirements.txt` | Python dependencies (CPU version of PyTorch) |
| `packages.txt` | System packages needed on the Streamlit server |

## Technologies Used
- Python
- Ultralytics YOLOv8
- EasyOCR
- OpenCV
- Streamlit and Gradio
- Google Colab

## Run It Locally
1. Clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open the address shown in the terminal and upload a vehicle photo.

To retrain the model, open `PlakaTahmin_1.ipynb` in Google Colab and follow the cells in order. The dataset zip file (`YOLO_TXT.zip`) is not included in this repository, so adjust the Google Drive paths in the notebook to your own files.

## Limitations
OCR accuracy depends on image quality, plate angle and lighting. Some characters, such as `T` and `I`, can still be confused. When the confidence is low, the app marks the result as "low confidence".

## Notes
This is a project I developed with the help of AI-assisted tools while learning Python and computer vision.
