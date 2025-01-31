import torch
from unsloth import FastVisionModel
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

def process_dataset_inference(model_name, dataset_name, output_csv_path):
    """Perform inference on Hugging Face dataset and compare results"""

    # Load the model and tokenizer from Unsloth
    model, tokenizer = FastVisionModel.from_pretrained(model_name)
    
    # Enable inference mode
    model = FastVisionModel.for_inference(model)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load dataset
    dataset = load_dataset(dataset_name)

    results = []

    # Iterate over dataset
    for item in dataset['train']:  # Adjust split if needed
        # Load image correctly
        if isinstance(item['image'], Image.Image):
            image = item['image']
        elif isinstance(item['image'], bytes):
            image = Image.open(io.BytesIO(item['image']))
        else:
            image = Image.open(item['image'])
        
        ground_truth_text = item['text']

        # Prepare input message
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe the handwritten text in this image into plain text."}
            ]}
        ]

        # Apply chat template correctly
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Ensure correct input format
        inputs = tokenizer(
            input_text,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                images=[image],  
                max_new_tokens=128,
                temperature=1.5,
                min_p=0.1
            )

        # Decode model output
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate CER & WER
        cer = calculate_cer(ground_truth_text, predicted_text)
        wer = calculate_wer(ground_truth_text, predicted_text)

        # Store results
        results.append({
            'image_path': item.get('image_path', 'Unknown'),
            'ground_truth': ground_truth_text,
            'model_output': predicted_text,
            'confidence_threshold': "N/A",  # Placeholder (modify if confidence is available)
            'CER': cer,
            'WER': wer
        })

    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    # Print summary
    print("📊 Inference Comparison Summary:")
    print(f"✅ Total Images Processed: {len(results)}")
    print(f"🔹 Average CER: {results_df['CER'].mean():.4f}")
    print(f"🔹 Average WER: {results_df['WER'].mean():.4f}")

# Run script
if __name__ == "__main__":
    process_dataset_inference(
        model_name="abhi26/QWEN2-2B-LITH_HT-lora_model",
        dataset_name="abhi26/LITH-HT1",
        output_csv_path="ocr_inference_results.csv"
    )
