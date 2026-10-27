# Turkish License Plate Recognition System

A Python application that automatically detects a license plate in a vehicle photo and reads the text on it. Plate detection uses a YOLOv8 model that I trained myself, and text reading uses EasyOCR. Results are shown in a simple web interface built with Gradio. The project was developed on Google Colab.

## How It Works
1. **Data preparation:** A labeled license plate dataset in YOLO format is loaded from Google Drive. All labels are converted to a single class ("plate").
2. **Model training:** A pretrained YOLOv8n model is trained on this dataset for plate detection. The best model is saved to Google Drive as `best_plaka_model.pt`, so training does not need to be repeated in later runs.
3. **Detection:** The saved model locates the plate in the photo and draws a bounding box around it.
4. **Reading:** The plate region is cropped. If it is small, it is enlarged and its contrast is increased, then EasyOCR reads the text.
5. **Correction:** The "TR" text is removed, and letters/digits that OCR commonly confuses (e.g. O/0, B/3) are corrected in the province code. The result is filtered with a regex that matches the Turkish plate format (such as `34ABC123`).

## Technologies Used
- Python
- Ultralytics YOLOv8
- EasyOCR
- OpenCV
- Matplotlib
- Gradio
- Google Colab

## How to Run
1. Open `PlakaTahmin_1.ipynb` in Google Colab.
2. Upload the dataset zip file (`YOLO_TXT.zip`) and the trained model file (`best_plaka_model.pt`) to your Google Drive. These files are not included in this repository, so adjust the paths in the notebook to match your own Drive.
3. Run the cells in order. `ultralytics`, `easyocr` and `gradio` are installed on the first run.
4. Upload a vehicle photo to the Gradio interface in the last section. The interface shows the image with the detected plate and the recognized plate text.

## Notes
This is a project I developed with the help of AI-assisted tools while learning Python. OCR results may be inaccurate depending on image quality.
