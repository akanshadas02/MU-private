import os
import torch
import pandas as pd
from unsloth import FastVisionModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

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
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
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
            if gt_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n] / max(m, n)

def process_dataset_inference(model_name, dataset_path, csv_file, output_csv_path):
    """Perform OCR inference on dataset and compare results"""
    
    # Load fine-tuned model & tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(model_name)
    model = FastVisionModel.for_inference(model)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load CSV file containing ground truth
    df = pd.read_csv(os.path.join(dataset_path, csv_file))

    results = []

    # Iterate over dataset
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        image_path = os.path.join(dataset_path, row['image_path'])
        ground_truth_text = row['text']

        if not os.path.exists(image_path):
            print(f"❌ Skipping {image_path} (File not found)")
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Prepare chat prompt correctly
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe the handwritten text in this image into plain text."}
                ]}
            ]

            # Apply chat template
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            # Tokenize input
            inputs = tokenizer(
                input_text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    images=[image],  # Ensure this is passed correctly
                    max_new_tokens=128,
                    temperature=1.5,
                    min_p=0.1
                )

            # Decode model output
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Compute CER & WER
            cer = calculate_cer(ground_truth_text, predicted_text)
            wer = calculate_wer(ground_truth_text, predicted_text)

            # Store results
            results.append({
                'image_path': image_path,
                'ground_truth': ground_truth_text,
                'model_output': predicted_text,
                'confidence_threshold': "N/A",  # Placeholder
                'CER': cer,
                'WER': wer
            })

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")

    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    # Print summary
    print("\n📊 Inference Completed!")
    print(f"✅ Total Images Processed: {len(results)}")
    print(f"🔹 Average CER: {results_df['CER'].mean():.4f}")
    print(f"🔹 Average WER: {results_df['WER'].mean():.4f}")

# Run script
if __name__ == "__main__":
    process_dataset_inference(
        model_name="abhi26/QWEN2-2B-LITH_HT-lora_model",
        dataset_path="path_to_dataset_folder",  # Change to your dataset path
        csv_file="dataset.csv",  # CSV filename containing image paths and ground truth
        output_csv_path="ocr_inference_results.csv"
    )
