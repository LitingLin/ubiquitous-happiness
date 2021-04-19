class OPEEvaluationParameter:
    bins_of_center_location_error = 51
    bins_of_normalized_center_location_error = 51
    bins_of_intersection_of_union = 21


def run_OPE_evalutation_and_generate_report(tracker_name, tracker, datasets, output_path, run_times=None, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    from evaluation.SOT.protocol.impl.ope_run_evalution import run_one_pass_evaluation
    run_one_pass_evaluation(tracker_name, tracker, datasets, output_path, run_times)
    from evaluation.SOT.protocol.impl.ope_report import generate_report_one_pass_evaluation
    generate_report_one_pass_evaluation(tracker_name, datasets, output_path, run_times, parameter)
