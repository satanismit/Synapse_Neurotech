import os
import glob
import random
from inference import InferencePipeline
from config import Config

def run_demo():
    # 1. Initialize
    print("Initializing Inference Pipeline...")
    try:
        pipeline = InferencePipeline(model_path=Config.MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("\n" + "="*50)
    print("   sEMG GESTURE RECOGNITION DEMO")
    print("="*50)
    print("Type 'exit' to quit.\n")

    while True:
        # 2. Get User Input
        print("\nOptions:")
        print("1. Paste the FULL PATH to your own .csv file (e.g., C:\\Users\\Name\\my_test_data.csv)")
        print("2. Press ENTER to test on a random file from the training set")
        print("3. Type 'exit' to quit")
        
        user_input = input("\n>> ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        target_file = None
        
        if not user_input:
            # Random mode
            search_path = os.path.join(Config.DATA_ROOT, "Session1", "session1_subject_1", "*.csv")
            files = glob.glob(search_path)
            if files:
                target_file = random.choice(files)
                print(f"Selected Random File: {target_file}")
            else:
                print("No files found for random selection.")
                continue
        else:
            # User mode
            # Remove quotes if user pasted path as "path"
            user_input = user_input.strip('"').strip("'")
            if os.path.exists(user_input) and user_input.endswith('.csv'):
                target_file = user_input
            else:
                print(f"Error: File not found or not a CSV: {user_input}")
                continue

        # 3. Predict
        print("Running prediction...")
        prediction = pipeline.predict(target_file)

        # 4. Result
        if prediction is not None:
            print(f"\n>>> PREDICTED GESTURE: {prediction}")
            
            # Try to infer true label from filename if possible
            basename = os.path.basename(target_file)
            if 'gesture' in basename:
                try:
                    true_label = int(basename.split('_')[0].replace('gesture', ''))
                    print(f">>> TRUE LABEL:      {true_label}")
                    if prediction == true_label:
                        print("✅ Correct!")
                    else:
                        print("❌ Incorrect")
                except:
                    pass
            print("-" * 30 + "\n")

if __name__ == "__main__":
    run_demo()
