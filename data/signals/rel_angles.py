import numpy as np

def get_relative_coordinate_angles(sample,
                                   references=(4, 8, 12, 16)):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((4*C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape

    rel_coords = []
    for i in range(len(references)):
        ref_loc = sample[:, :, references[i], :]
        coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
        rel_coords.append(coords_diff)

    rel_angles = []
    for coords in rel_coords:
        flattenx = coords[0, :, :, :].reshape((t * V * M))
        flatteny = coords[1, :, :, :].reshape((t * V * M))
        flattenz = coords[2, :, :, :].reshape((t * V * M))
        odxy = np.arctan2(flatteny, flattenx + 1e-10) * (180 / np.pi)
        odyz = np.arctan2(flattenz, flatteny + 1e-10) * (180 / np.pi)
        odxz = np.arctan2(flattenz, flattenx + 1e-10) * (180 / np.pi)

        xy = odxy.reshape((t, V, M))
        yz = odyz.reshape((t, V, M))
        xz = odxz.reshape((t, V, M))
        rel_angles.append(np.stack([xy, yz, xz]))

    # Shape: 4*C, t, V, M
    rel_angles = np.vstack(rel_angles)

    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_angles
    return final_sample