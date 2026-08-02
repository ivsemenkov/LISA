"""Global constants for data paths and dataset metadata."""

import os


AUDIO_SR = 16000
# Embedding size for wav2vec2-base (the only model used in training)
N_FEATURES = 768

# Training constants (hardcoded, not configurable via CLI)
CHECKPOINT_NAME = 'lisa_checkpoint'
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
CLEAN_DATA_DIR = os.path.join(DATA_ROOT, 'clean')
PREPROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'preprocessed')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
EXPERIMENTS_DIR = os.path.join(OUTPUTS_DIR, 'experiments')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
OCCLUSION_FEATURE_ANALYSIS_DIR = os.path.join(
    PLOTS_DIR,
    'occlusion_feature_analysis',
)
BRANCH_INTERPRETATION_ASSIGNMENTS_DIR = os.path.join(OUTPUTS_DIR, 'motif_clusters')

# Dataset constants
SUB_SES_COMBOS = [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (3, 0),
    (4, 0),
    (4, 1),
    (5, 0),
    (5, 1),
    (6, 0),
    (6, 1),
    (7, 0),
    (7, 1),
    (8, 0),
    (8, 1),
    (9, 0),
    (9, 1),
    (10, 0),
    (10, 1),
    (11, 0),
    (11, 1),
    (12, 0),
    (13, 0),
    (13, 1),
    (14, 0),
    (14, 1),
    (15, 0),
    (15, 1),
    (16, 0),
    (17, 0),
    (17, 1),
    (18, 0),
    (18, 1),
    (19, 0),
    (19, 1),
    (20, 0),
    (21, 0),
    (22, 0),
    (22, 1),
    (23, 0),
    (23, 1),
    (24, 0),
    (24, 1),
    (25, 0),
    (25, 1),
    (26, 0),
    (26, 1),
    (27, 0),
    (27, 1),
]
N_MEG_CHANNELS = 208
N_SUBJECTS = len(set(sub_ses[0] for sub_ses in SUB_SES_COMBOS))
STORY_IDS = [0, 1, 2, 3]
TRAIN_STORIES = {
    0: ('lw1', 0, 3),
    1: ('cable_spool_fort', 0, 5),
    2: ('easy_money', 0, 7),
    3: ('the_black_willow', 0, 4),
}
TEST_STORIES = {
    3: ('the_black_willow', 5, 11),
}
