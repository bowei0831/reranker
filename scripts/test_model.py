from FlagEmbedding import FlagReranker

def main():
    model_path = "/home/peter831/test/outputs_large/large_bge_add_library"

    print("載入模型...")
    reranker = FlagReranker(model_path, use_fp16=True)

    print("測試推論...")
    score = reranker.compute_score(["今天天氣如何", "今天是晴天，氣溫25度"])
    print(f"分數: {score}")

    print("模型載入成功！")

if __name__ == '__main__':
    main()