import numpy as np

def get_oriented_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    person1 = disps[:, :, :, 0]
    person2 = disps[:, :, :, 1]
    cog1 = person1.mean(axis=2).mean(axis=1)
    cog2 = person2.mean(axis=2).mean(axis=1)

    person1 = (person1.transpose(2, 1, 0) - cog1).transpose((2, 1, 0))
    person2 = (person2.transpose(2, 1, 0) - cog2).transpose((2, 1, 0))
    disps = np.stack([person1, person2], axis=3)
    
    flattenx = disps[0, :, :, :].reshape(((t-1) * V * M))
    flatteny = disps[1, :, :, :].reshape(((t-1) * V * M))
    flattenz = disps[2, :, :, :].reshape(((t-1) * V * M))
    odxy = np.arctan2(flatteny, flattenx + 1e-10) * (180 / np.pi)
    odyz = np.arctan2(flattenz, flatteny + 1e-10) * (180 / np.pi)
    odxz = np.arctan2(flattenz, flattenx + 1e-10) * (180 / np.pi)

    xy = odxy.reshape(((t-1), V, M))
    yz = odyz.reshape(((t-1), V, M))
    xz = odxz.reshape(((t-1), V, M))

    # Shape: C, t-1, V, M
    orient_disps = np.stack([xy, yz, xz])
    # Shape C, T, V, M
    final_sample[:, start:end-1, :, :] = orient_disps

    return final_sample
