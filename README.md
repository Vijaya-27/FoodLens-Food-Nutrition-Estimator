# FoodLens-Food-Nutrition-Estimator

A Deep Learning-based web application built with **TensorFlow** and **Streamlit** that predicts the nutritional values of a food image. 

## Features

- Upload a food image
- Predict nutritional values using a CNN model
- Display:
  - Calories (kcal)
  - Protein (g)
  - Carbohydrates (g)
  - Fat (g)
- Automatic model training if no trained model exists
- Interactive Streamlit interface
- TensorFlow/Keras implementation

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Pillow (PIL)

---

## Project Structure

```
Food-Nutrition-Detection/

│── app.py
│── Food_Nutrition_Dataset.csv
│── nutrition_model.keras
│── nutrition_model.keras.norm.json
│── requirements.txt
│── README.md
```

---

## Dataset

The project uses:

**Food_Nutrition_Dataset.csv**

Target columns:

- Calories
- Protein
- Carbs
- Fat

Since the dataset does not contain food images, the current implementation generates synthetic images during training.

---

## Run the Application

```bash
streamlit run app.py
```

---

## Application Workflow

1. Launch the Streamlit application.
2. Load or train the CNN model.
3. Upload a food image.
4. The image is preprocessed and passed through the trained model.
5. The model predicts:
   - Calories
   - Protein
   - Carbohydrates
   - Fat
6. Results are displayed in an interactive dashboard.

---

## Future Improvements

- Train using real food image datasets.
- Integrate EfficientNetB0 or ResNet50 using Transfer Learning.
- Food classification before nutrition estimation.
- Portion size estimation.
- Health score calculation.
- Nutrition charts and visualizations.
- Meal history tracking.
- Multiple food detection in a single image.

---
