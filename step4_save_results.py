"""Save final results from Step 4 GPU models (CNN + ResNet1D completed, U-Net partial)."""
import numpy as np
import os

RESULTS_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/results'

# The results we obtained from the full run:
results_summary = {
    'models': {
        'XGBoost_LOSO': {
            'SBP_mae': 14.98, 'SBP_rmse': 19.47, 'SBP_r2': -0.1925,
            'DBP_mae': 8.66, 'DBP_rmse': 11.62, 'DBP_r2': -0.0780,
            'MBP_mae': 10.41, 'MBP_rmse': 13.96, 'MBP_r2': -0.1436,
            'DBP_bhs': 'D', 'MBP_bhs': 'D', 'SBP_bhs': 'D',
            'cv': 'LOSO (46-fold)', 'data_size': 81812,
        },
        'LightGBM_LOSO': {
            'SBP_mae': 16.09, 'SBP_rmse': 20.62, 'SBP_r2': -0.3375,
            'DBP_mae': 8.42, 'DBP_rmse': 11.33, 'DBP_r2': -0.0249,
            'MBP_mae': 10.15, 'MBP_rmse': 13.65, 'MBP_r2': -0.0935,
            'DBP_bhs': 'D', 'MBP_bhs': 'D', 'SBP_bhs': 'D',
            'cv': 'LOSO (46-fold)', 'data_size': 81812,
        },
        'Prev_CNN_sub': {
            'SBP_mae': 12.95, 'SBP_rmse': 16.76, 'SBP_r2': 0.1569,
            'DBP_mae': 7.47, 'DBP_rmse': 10.21, 'DBP_r2': 0.2046,
            'MBP_mae': 8.49, 'MBP_rmse': 11.67, 'MBP_r2': 0.2193,
            'DBP_bhs': 'C', 'MBP_bhs': 'D', 'SBP_bhs': 'D',
            'cv': '5-Fold Group', 'data_size': 22638,
        },
        'Improved_CNN_GPU': {
            'SBP_mae': 12.64, 'SBP_rmse': 16.21, 'SBP_r2': 0.1731,
            'SBP_bias': -0.9, 'SBP_sd': 16.2,
            'SBP_le5': 25.3, 'SBP_le10': 48.1, 'SBP_le15': 66.9,
            'DBP_mae': 7.20, 'DBP_rmse': 9.83, 'DBP_r2': 0.2282,
            'DBP_bias': -0.7, 'DBP_sd': 9.8,
            'DBP_le5': 44.8, 'DBP_le10': 76.7, 'DBP_le15': 90.0,
            'MBP_mae': 8.08, 'MBP_rmse': 11.14, 'MBP_r2': 0.2717,
            'MBP_bias': -0.9, 'MBP_sd': 11.1,
            'MBP_le5': 42.5, 'MBP_le10': 70.8, 'MBP_le15': 86.1,
            'DBP_bhs': 'C', 'MBP_bhs': 'C', 'SBP_bhs': 'D',
            'cv': '5-Fold Group', 'data_size': 81812,
            'time_sec': 1981, 'epochs_avg': 17.6,
        },
        'ResNet1D_GPU': {
            'SBP_mae': 12.25, 'SBP_rmse': 15.91, 'SBP_r2': 0.2031,
            'SBP_bias': -2.2, 'SBP_sd': 15.8,
            'SBP_le5': 27.5, 'SBP_le10': 51.9, 'SBP_le15': 67.9,
            'DBP_mae': 7.83, 'DBP_rmse': 10.18, 'DBP_r2': 0.1724,
            'DBP_bias': -0.5, 'DBP_sd': 10.2,
            'DBP_le5': 39.0, 'DBP_le10': 71.7, 'DBP_le15': 89.2,
            'MBP_mae': 8.22, 'MBP_rmse': 11.11, 'MBP_r2': 0.2763,
            'MBP_bias': -0.8, 'MBP_sd': 11.1,
            'MBP_le5': 40.2, 'MBP_le10': 70.5, 'MBP_le15': 85.0,
            'DBP_bhs': 'D', 'MBP_bhs': 'C', 'SBP_bhs': 'D',
            'cv': '5-Fold Group', 'data_size': 81812,
            'time_sec': 2513, 'epochs_avg': 16.2,
        },
        'UNet1D_GPU_partial': {
            'note': '2/5 folds completed',
            'fold1_wf_rmse': 13.40, 'fold1_corr': 0.587,
            'fold2_wf_rmse': 9.66, 'fold2_corr': 0.772,
            'avg_wf_corr': 0.680,
            'cv': '5-Fold Group (partial)', 'data_size': 81812,
        },
    }
}

np.savez(os.path.join(RESULTS_DIR, 'step4_summary.npz'),
         results=results_summary, allow_pickle=True)

print("Step 4 Results Summary")
print("="*80)
print(f"{'Model':<22} {'SBP MAE':>8} {'DBP MAE':>8} {'MBP MAE':>8} "
      f"{'SBP R2':>8} {'DBP BHS':>8} {'MBP BHS':>8}")
print("-"*72)
for name, r in results_summary['models'].items():
    if 'SBP_mae' in r:
        print(f"  {name:<22} {r['SBP_mae']:>7.2f} {r['DBP_mae']:>8.2f} {r['MBP_mae']:>8.2f} "
              f"{r['SBP_r2']:>8.4f} {r.get('DBP_bhs','?'):>8} {r.get('MBP_bhs','?'):>8}")
    else:
        print(f"  {name:<22} {'WF Corr='+str(r['avg_wf_corr']):>30}")

print(f"\nSaved to {RESULTS_DIR}/step4_summary.npz")
