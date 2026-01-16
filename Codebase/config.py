import os

class Config:
    # Paths
    DATA_ROOT = r"d:/Hackathon/synapse/Synapse_Dataset/Synapse_Dataset"
    OUTPUT_DIR = "output"
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # Data
    SESSIONS = ["Session1", "Session2", "Session3"]
    NUM_CHANNELS = 8
    SAMPLE_RATE = 512
    DURATION = 5  # seconds
    WINDOW_SIZE = SAMPLE_RATE * DURATION  # 2560
    NUM_CLASSES = 5
    
    # Preprocessing
    LOWCUT = 20.0
    HIGHCUT = 200.0 # Lowered from 450 to 200 to be safe and focus on main EMG power
    NOTCH_FREQ = 50.0
    FILTER_ORDER = 4

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10
    SEED = 42
    
    # Split
    VAL_SUBJECTS = [21, 22, 23, 24, 25] # Last 5 subjects for validation
    TRAIN_SUBJECTS = [i for i in range(1, 26) if i not in [21, 22, 23, 24, 25]]

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
