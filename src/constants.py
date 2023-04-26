import os

ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR+"/data"
ALLOWED_ALGO = {
    "DQN": "Deep Q Network",
    "DDQN": "Double Deep Q Network"
}
SEED_VALUE=50
STATE_SIZE = 10 # Window size: representing number of items seen in current window
