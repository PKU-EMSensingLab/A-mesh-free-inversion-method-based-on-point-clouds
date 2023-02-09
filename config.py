cfg = {
    'MODEL_SAVE_PATH': './Model/',
    'DATA_ROOT': './Dataset/',
    'FIG_ROOT': './Fig/',
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 125,
    'EPOCHS': 500,
    'NUM_COUPLING_LAYERS': 4,
    'NUM_NET_LAYERS': 6,  # neural net layers for each coupling layer
    'POINT_NUM': 1000,
    'HIDDEN_DIM': 128,
    'FIELD_DIM': 216,
    'PC_DIM': 4,
    'WEIGHT': 1.0,
    'SNAPSHOT': 50,
    'LATENT_DIM': 64,
    'TRAIN_NUM': 5000,
    'DEV_NUM': 500,
    'DOI_MIN': -0.6,
    'DOI_MAX': 0.6,
    'LOAD': False
}