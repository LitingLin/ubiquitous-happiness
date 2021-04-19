from evaluation.SOT.sample_tracker import IdentityTracker
from evaluation.SOT.runner import run_standard_evaluation


if __name__ == '__main__':
    run_standard_evaluation('test', IdentityTracker(), 'C:\\test\\')
