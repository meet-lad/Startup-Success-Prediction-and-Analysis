# Startup Success Predictor

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
python src/train_model.py
streamlit run Streamlit_app/app.py
```

The app accepts a startup's age, total funding, funding rounds, industry, and country, then estimates the chance of success. The training script saves the complete preprocessing-and-model pipeline in `Models/`, so the deployed app uses the same transformations as training.

## Deploy on Streamlit Community Cloud

Push this repository to GitHub, create a new Streamlit app, and set the main file path to `Streamlit_app/app.py`. Streamlit installs the root `requirements.txt`. If the generated files in `Models/` are not yet committed, the app automatically trains and saves them from `Data/processed/processed_dataset.csv` on its first start.
