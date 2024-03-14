from simca import load_yaml_config, save_config_file
import numpy as np
import os

file_list = ["predict_results_recons.yml", "predict_results_full_learned_mask.yml", "predict_results_full_learned_mask_float.yml"]

for file in file_list:
    if not os.path.exists(f'./results/{file}'):
        continue
    dict = load_yaml_config(f'./results/{file}')

    ssim_array = [0 for i in range(20)]
    rmse_array = [0 for i in range(20)]
    psnr_array = [0 for i in range(20)]

    list_rmse = []
    list_ssim = []
    list_psnr = []

    for key in dict.keys():
        list_ = dict[key]
        val = np.array(list_)
        if all(char.isdigit() for char in key[-2:]):
            if 'SSIM' in key:
                ssim_array[int(key[-2:])-1] = float(np.mean(val))
                list_ssim = list_ssim + list_
            elif 'RMSE' in key:
                rmse_array[int(key[-2:])-1] = float(np.mean(val))
                list_rmse = list_rmse + list_
            elif 'PSNR' in key:
                psnr_array[int(key[-2:])-1] = float(np.mean(val))
                list_psnr = list_psnr + list_
        else:
            if 'SSIM' in key:
                ssim_array[int(key[-1])-1] = float(np.mean(val))
                list_ssim = list_ssim + list_
            elif 'RMSE' in key:
                rmse_array[int(key[-1])-1] = float(np.mean(val))
                list_rmse = list_rmse + list_
            elif 'PSNR' in key:
                psnr_array[int(key[-1])-1] = float(np.mean(val))
                list_psnr = list_psnr + list_
            
    array_list_ssim = np.array(list_ssim)
    array_list_rmse = np.array(list_rmse)
    array_list_psnr = np.array(list_psnr)

    array_list_ssim = array_list_ssim[np.nonzero(array_list_ssim)]
    array_list_rmse = array_list_rmse[np.nonzero(array_list_rmse)]
    array_list_psnr = array_list_psnr[np.nonzero(array_list_psnr)]

    res_dict = {'SSIM': ssim_array,
                'SSIM overall': float(np.mean(array_list_ssim)),
                'RMSE': rmse_array,
                'RMSE overall': float(np.mean(array_list_rmse)),
                'PSNR': psnr_array,
                'PSNR overall': float(np.mean(array_list_psnr))}

    print("SSIM std: ", np.std(np.array(list_ssim)))
    print("RMSE std: ", np.std(np.array(list_rmse)))
    print("PSNR std: ", np.std(np.array(list_psnr)))

    save_config_file("mean_"+file[:-4], res_dict, './results')