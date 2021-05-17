import torch
import os
import shutil


def _fail_safe_save(path, state_dict, overwrite_existing=True):
    torch.save(state_dict, path + '.tmp')
    if overwrite_existing and os.path.exists(path):
        os.remove(path)
    os.rename(path + '.tmp', path)


def _fail_safe_copy(src_path, dst_path, overwrite_existing=True):
    shutil.copy(src_path, dst_path + '.tmp')
    if overwrite_existing and os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(dst_path + '.tmp', dst_path)


def dump_checkpoint(epoch, output_path, state_dict, latest_checkpoint_interval, backup_checkpoint_interval):
    saved_checkpoint_path = None
    if (epoch + 1) % latest_checkpoint_interval == 0:
        checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
        _fail_safe_save(checkpoint_path, state_dict)
        saved_checkpoint_path = checkpoint_path
    # extra checkpoint before LR drop and every 100 epochs
    if (epoch + 1) % backup_checkpoint_interval == 0:
        backup_checkpoint_path = os.path.join(output_path, f'checkpoint{epoch:04}.pth')
        if saved_checkpoint_path is not None:
            _fail_safe_copy(saved_checkpoint_path, backup_checkpoint_path)
        else:
            _fail_safe_save(backup_checkpoint_path, state_dict)
            saved_checkpoint_path = backup_checkpoint_path
