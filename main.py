import uvicorn
from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import pickle
from typing import Optional
import pandas as pd 
import pandas as pd 
import pandas as pd 
import pandas as pd 
# loading the model
model = pickle.load(open('admission_prediction_model.pkl', 'rb'))

# making the standard scaler
df = pd.read_csv('./data/Admission_Prediction_Data.csv')
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
sc = StandardScaler()
X = sc.fit_transform(X)

app = FastAPI()
templates = Jinja2Templates(directory="templates") # for html templates
app.mount('/static', StaticFiles(directory="static"), name="static") # for static(css) files

@app.get('/')
def predict_function(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/', response_class=HTMLResponse)
def predict_function(request: Request,
                    gre_score: float = Form(...),
                    toefl_score: float = Form(...),
                    university_rating: float = Form(...),
                    sop_score: float = Form(...),
                    lor_score: float = Form(...),
                    cgpa_score: float = Form(...),
                    research: str = Form(...)
                    ):
    
    research_score = 0
    if(research == 'Yes'):
        research_score = 1
    
    # standardizing the data before passing to the model
    sample = sc.transform([[gre_score, toefl_score, university_rating, sop_score, lor_score, cgpa_score, research_score]])
    prediction = f"Chances of Admission: {model.predict(sample) * 100}%"
    return templates.TemplateResponse('result.html', {'request': request, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)