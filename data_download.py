import kagglehub

# Download latest version
path = kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis", path="")

print("Path to dataset files:", path)