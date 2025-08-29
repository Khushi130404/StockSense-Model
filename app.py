# # app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
# import model
#  # your training + get_prediction function

# app = FastAPI()

# class PredictRequest(BaseModel):
#     ticker: str

# @app.post("/predict")
# def predict(req: PredictRequest):
#     result = my_model.get_prediction(req.ticker)
#     return result

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
