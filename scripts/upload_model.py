from huggingface_hub import HfApi, create_repo

# 設定
model_path = "/home/peter831/test/outputs_large/large_bge_add_library"
repo_name = "peter831/reranker-large-v1"  # 改成你的 username

# 建立 repo（如果不存在）
api = HfApi()
try:
    create_repo(repo_name, repo_type="model", exist_ok=True)
    print(f"Repo 建立成功: {repo_name}")
except Exception as e:
    print(f"Repo 已存在或錯誤: {e}")

# 上傳
print("開始上傳...")
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_name,
    repo_type="model",
)
print(f"上傳完成！")
print(f"模型網址: https://huggingface.co/{repo_name}")
