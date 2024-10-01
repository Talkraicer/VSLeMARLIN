# Paths
ROOT = r"C:\Users\tal\OneDrive - Technion\Studies\Eyal Project"
RESULTS_FOLDER = r"C:/Eyal Project"

# Experiment Settings
EXP_NAME = "VSL"
DEMANDS = ['low', 'medium', 'high']

SEGMENTS = [f'Fi{i}' for i in range(1, 6)] + \
            [f'Mo{i}' for i in range(1, 6)] + \
            [f'Mi{i}+S{i}' for i in range(1, 6)] + \
            [f'E{i}' for i in range(1, 6)]


# Machine Settings
GUI = False
NUM_PROCESSES = 10

# Randomization Settings
SEED = 42