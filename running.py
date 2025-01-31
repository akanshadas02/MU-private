import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from PIL import Image
import io

def calculate_cer(ground_truth, predicted):
    """Calculate Character Error Rate (CER)"""
    gt, pred = ground_truth.lower(), predicted.lower()
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, n)

def calculate_wer(ground_truth, predicted):
    """Calculate Word Error Rate (WER)"""
    gt_words, pred_words = ground_truth.lower().split(), predicted.lower().split()
    m, n = len(gt_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_words[i-1] == pred_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, n)

def process_dataset_inference(model_name, dataset_name, output_csv_path):
    """Perform OCR inference and evaluation on Hugging Face dataset"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    dataset = load_dataset(dataset_name)
    results = []
    
    for item in dataset['train']:
        image = Image.open(io.BytesIO(item['image'])) if isinstance(item['image'], bytes) else item['image']
        ground_truth_text = item['text']
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe the handwritten text in this image into a Plain Text-formatted representation."}
            ]}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, images=[image], max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        confidence = float(torch.mean(torch.softmax(outputs[0], dim=-1)).cpu().numpy())
        
        cer = calculate_cer(ground_truth_text, predicted_text)
        wer = calculate_wer(ground_truth_text, predicted_text)
        
        results.append({
            'image_path': item.get('image_path', 'Unknown'),
            'ground_truth': ground_truth_text,
            'model_output': predicted_text,
            'confidence_threshold': confidence,
            'CER': cer,
            'WER': wer
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    
    print("Inference Summary:")
    print(f"Total Images Processed: {len(results)}")
    print(f"Average CER: {results_df['CER'].mean():.4f}")
    print(f"Average WER: {results_df['WER'].mean():.4f}")

if __name__ == "__main__":
    process_dataset_inference(
        model_name="abhi26/QWEN2-2B-LITH_HT-lora_model",
        dataset_name="abhi26/LITH-HT1",
        output_csv_path="ocr_inference_results.csv"
    )
