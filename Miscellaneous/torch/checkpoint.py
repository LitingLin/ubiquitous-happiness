import torch
import os
import shutil
import pickle


def _get_training_state_file_path(model_state_file_path: str):
    return os.path.dirname(model_state_file_path) + '-training.pkl'


def _safe_rename(model_state_file_path, training_state_file_path, overwrite_existing=True):
    if overwrite_existing and os.path.exists(model_state_file_path):
        os.remove(model_state_file_path)
    if overwrite_existing and os.path.exists(training_state_file_path):
        os.remove(training_state_file_path)
    os.rename(model_state_file_path + '.tmp', model_state_file_path)
    os.rename(training_state_file_path + '.tmp', training_state_file_path)


def _fail_safe_save(path, model_state_dict, training_state_dict, overwrite_existing=True):
    training_state_file_path = _get_training_state_file_path(path)
    torch.save(model_state_dict, path + '.tmp')
    with open(training_state_file_path + '.tmp', 'wb') as f:
        pickle.dump(training_state_dict, f)
    _safe_rename(path, training_state_file_path, overwrite_existing)


def _fail_safe_copy(src_path, dst_path, overwrite_existing=True):
    src_training_state_file_path = _get_training_state_file_path(src_path)
    dst_training_state_file_path = _get_training_state_file_path(dst_path)
    shutil.copy(src_path, dst_path + '.tmp')
    shutil.copy(src_training_state_file_path, dst_training_state_file_path + '.tmp')
    _safe_rename(dst_path, dst_training_state_file_path, overwrite_existing)


def dump_checkpoint(epoch, output_path, model_state_dict, training_state_dict, latest_checkpoint_interval, backup_checkpoint_interval):
    saved_checkpoint_path = None
    if (epoch + 1) % latest_checkpoint_interval == 0:
        checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
        _fail_safe_save(checkpoint_path, model_state_dict, training_state_dict)
        saved_checkpoint_path = checkpoint_path
    # extra checkpoint before LR drop and every 100 epochs
    if (epoch + 1) % backup_checkpoint_interval == 0:
        backup_checkpoint_path = os.path.join(output_path, f'checkpoint{epoch:04}.pth')
        if saved_checkpoint_path is not None:
            _fail_safe_copy(saved_checkpoint_path, backup_checkpoint_path)
        else:
            _fail_safe_save(backup_checkpoint_path, model_state_dict, training_state_dict)
            saved_checkpoint_path = backup_checkpoint_path


def load_checkpoint(checkpoint_path: str):
    training_state_file_path = _get_training_state_file_path(checkpoint_path)
    model_state_dict = torch.load(checkpoint_path, map_location='cpu')
    with open(training_state_file_path, 'rb') as f:
        training_state_dict = pickle.load(f)
    return model_state_dict, training_state_dict
