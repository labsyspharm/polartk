import numpy as np

import polartk

def run_job(job: np.ndarray) -> np.ndarray:
    # params
    params = {
        'pw': 1, # pad width, unit: pixel
        'pv': 0, # pad value for image
        'n_neighbors': 5, # params for KNN intensity model
    }

    # unpack job
    label_xy = job[..., 0]
    images = [job[..., i] for i in range(1, job.shape[2])]

    # run transformation
    out_dict = polartk.xy2rt(images=images, label=label_xy)

    # pack output
    output_array = np.zeros_like(job)
    output_array[..., 0] = out_dict['label_rt']
    for i, image_rt in enumerate(out_dict['image_rt_list']):
        output_array[..., i+1] = image_rt

    return output_array

if __name__ == '__main__':
    # paths
    input_filepath = './job_1497.npy'
    output_filepath = './output_job_1497.npy'

    # run job
    job = np.load(input_filepath)
    output = xy2rt(job)
    np.save(output_filepath, output)

