import kagglehub

# Download latest version
path = kagglehub.dataset_download("sumitm004/arxiv-scientific-research-papers-dataset")

print("Path to dataset files:", path)