import numpy as np
from deephistopath.evaluation import evaluate_global_f1
import deephistopath.evaluation as ev
import deephistopath.detection as det
import shutil
img_dir =  "data/mitoses/mitoses_train_image_data"
#pred_dir = "pred/result/2018/new_orig_v3_fp_v3_8_icpr/scratch/resnet_custom/resnet_custom_stride_32_nm_model_180307_175241"
#pred_dir = "pred/result/2018/new_orig_v3_fp_v3_8_icpr/scratch/resnet_custom/resnet_custom_stride_16_nm_model_180307_175241"
#pred_dir = "pred/result/2018/new_orig_v3_fp_v3_8_icpr/scratch/resnet_custom/resnet_custom_stride_1_nm_model_180307_175241"
#pred_dir = "pred/result/2018/new_orig_v3_fp_v3_8_icpr/scratch/resnet_custom/resnet_custom_stride_16_marg_model_180307_175241"
#ground_true_dir = "pred/data/val_ground_truth"
pred_dir = "/8tb/deep_histopath/result/mitoses/mitoses_test_image_result"

print("raw predictions:")
threshold = 30
prob_thresh = 0.5
f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(pred_dir,
    ground_true_dir, threshold, prob_thresh)
print(f"F1: {f1}; Precision: {ppv}; Recall: {sens}; FP: {len(fp_list)}; TP: {len(tp_list)}; FN: "\
      f"{len(fn_list)}")

# smooth prob maps
print("smoothing predictions:")
radius = 15
det.smooth_prediction_results(pred_dir, img_dir, radius, True)

# NOTE: this is probably not what we want to evaluate, because the smoothing algorithm needs the
# ensuing detection step to be useful
print("smoothed predictions:")
smooth_pred_dir = pred_dir + "_smoothed"
#threshold = 30  # 27
#prob_thresh = 0.35 #65
#f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(
#    smooth_pred_dir, ground_true_dir, threshold, prob_thresh)
#print(f"F1: {f1}; Precision: {ppv}; Recall: {sens}; FP: {len(fp_list)}; TP: {len(tp_list)}; FN: "\
#      f"{len(fn_list)}")

# detect mitoses
prob_thresh = 0.35
det.detect_prediction_results(smooth_pred_dir, img_dir, radius, prob_thresh, True)

print("detected smoothed predictions:")
detected_pred_dir = smooth_pred_dir + "_detected"
threshold = 30
prob_thresh = 0.65  #prob_thresh
f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(
    detected_pred_dir, ground_true_dir, threshold, prob_thresh)
print(f"F1: {f1}; Precision: {ppv}; Recall: {sens}; FP: {len(fp_list)}; TP: {len(tp_list)}; FN: "\
      f"{len(fn_list)}")

# ---
# compare to clustering on the original probs
# NOTE: if this is re-run, be sure to remove the existing directory prior to running
#if os.path.exists(clustered_pred_dir)
#shutil.rmtree(clustered_pred_dir)
#`rm -rf pred/result/2018/new_orig_v3_fp_v3_8_icpr/scratch/resnet_custom/resnet_custom_stride_16_marg_model_180307_175241_clustered/`
eps = 60  # radius from center is 7.5 µm = 30 pixels, so diameter = 60 pixels
prob_thresh = 0.4
det.cluster_prediction_result(
    pred_dir, eps=eps, min_samples=2, hasHeader=True, isWeightedAvg=False,
    prob_threshold=prob_thresh)

print("clustered raw predictions:")
clustered_pred_dir = pred_dir + "_clustered"
threshold = 30
prob_thresh = 0.59
f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(
    clustered_pred_dir, ground_true_dir, threshold, prob_thresh)
print(f"F1: {f1}; Precision: {ppv}; Recall: {sens}; FP: {len(fp_list)}; TP: {len(tp_list)}; FN: "\
      f"{len(fn_list)}")

