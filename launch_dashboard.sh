echo "Running on host: $(hostname) / $(hostname -i)"

uv run streamlit run src/dashboard/app.py --server.address 0.0.0.0
