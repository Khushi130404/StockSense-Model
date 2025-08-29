# Stock-Sense Model

- python -m venv venv
- .\venv\Scripts\Activate.ps1
- pip install -r requirements.txt
- python src/etl_pipeline.py
- python src/model.py --db_path data/stock_data.db --ticker AAPL --epochs 50 --batch_size 32
- uvicorn api.main:app --reload --port 8001
- http://127.0.0.1:8001/docs
