from tensorflow.keras.callbacks import \
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os


def getCallbacks(patience, save_root, batch_size):
    # Define saving file paths
    save_model_path = "{}/model.h5".format(save_root)
    csv_log_file_name = "{}/csv_log_file.csv".format(save_root)

    # Create folders if do not exist
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    # Define callbacks
    early_stopping = EarlyStopping(patience=patience)
    checkpointer = ModelCheckpoint(
        save_model_path, verbose=1, save_best_only=True)
    reduce_learning_rate = ReduceLROnPlateau(
        factor=0.3, patience=4, min_lr=1e-8)
    csv_logger = CSVLogger(csv_log_file_name, separator=';')
    tensor_board = TensorBoard(
        log_dir=save_root, write_graph=False, batch_size=batch_size)
    callbacks = [
        early_stopping,
        checkpointer,
        reduce_learning_rate,
        csv_logger,
        tensor_board]
    return callbacks
