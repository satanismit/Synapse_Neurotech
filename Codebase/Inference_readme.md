# Inference Pipeline Execution Guide

This guide details the steps to execute the inference pipeline for sEMG gesture classification.

## Prerequisites
- Ensure `python` is installed and added to PATH.
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Ensure the trained model exists at `output/best_model.pth`.

## Method 1: Interactive Demo (Recommended)
This method allows you to interactively test the model on random files or specific files.

**Steps:**
1. Navigate to the `Codebase` directory:
   ```cmd
   cd d:\Hackathon\synapse\Synapse_Dataset\Synapse_Dataset\Codebase
   ```
2. Run the batch file:
   ```cmd
   run_inference.bat
   ```
   *Alternatively, run the python script directly:*
   ```cmd
   python Inference_pipeline.py
   ```
3. Follow the on-screen prompts:
   - **Option 1**: Paste the full path to a CSV file (e.g., `C:\Data\test_gesture.csv`) to predict its label.
   - **Option 2**: Press `ENTER` to let the script pick a random file from the dataset for testing.
   - **Option 3**: Type `exit` to close the tool.

## Method 2: Command Line Interface (CLI)
This method is useful for automated scripts or single-file predictions without interactivity.

**Steps:**
1. Open a terminal/command prompt.
2. Run the `inference.py` script with the target file path:
   ```cmd
   python inference.py "path/to/your/input_file.csv"
   ```
   *Example:*
   ```cmd
   python inference.py "d:/Hackathon/synapse/Synapse_Dataset/Synapse_Dataset/Dataset/Session1/session1_subject_1/gesture00_trial01.csv"
   ```
3. The script will output the predicted gesture label.

## Method 3: Bulk Inference
To predict gestures for ALL files in a directory:

**Steps:**
1. Run `inference.py` pointing to a folder:
   ```cmd
   python inference.py "path/to/folder_containing_csvs/"
   ```
2. The script will list predictions for every CSV file found in that folder.

## Troubleshooting
- **Model not found**: Ensure `output/best_model.pth` exists. If not, run `python train.py` first.
- **File not found**: Check if the path you pasted is correct and wrapped in quotes if it contains spaces.
- **Import errors**: Run `pip install -r requirements.txt` again.
