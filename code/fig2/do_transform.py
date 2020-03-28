import os
import sys
code_folderpath = os.path.expanduser('~/polartk/code/')
sys.path.append(code_folderpath)

import numpy as np
import tqdm

import polartk

if __name__ == '__main__':
    data_folderpath = './scenario_data'
    scenario_list = []
    for name in os.listdir(data_folderpath):
        if os.path.splitext(name)[1] == '.npy' and not name.startswith('output_'):
            scenario_list.append(name)

    for name in tqdm.tqdm(scenario_list):
        input_filepath = os.path.join(data_folderpath, name)
        output_filepath = os.path.join(data_folderpath, 'output_'+name)
        job = np.load(input_filepath)
        _, _, image_rt, label_rt = polartk.xy2rt(image=job[..., 1],
                label=job[..., 0])
        output = np.stack([label_rt, image_rt], axis=-1)
        np.save(output_filepath, output)

