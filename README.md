# Stock-Sense Model

- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- python src/etl_pipeline.py
- python src/model.py --db_path data/stock_data.db --ticker AAPL --epochs 50 --batch_size 32