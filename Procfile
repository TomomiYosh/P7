api: gunicorn scoringapi:app -b 0.0.0.0:$PORT
web: streamlit run dashbord.py --server.port $PORT