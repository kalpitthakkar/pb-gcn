import numpy as np

def RandomTemporalCrop(**kwargs):
    sample = kwargs['sample']
    size = kwargs['window_size']

    T = sample.shape[1]

    if T <= size:
        return sample
    else:
        start = np.random.randint(T - size)
        return sample[:, start:start + size, :, :]

def RandomTemporalSampling(**kwargs):
    sample = kwargs['sample']
    num_samples = kwargs['num_samples']

    T = sample.shape[1]

    if num_samples >= T:
        return sample

    idx = np.random.choice(T, num_samples, replace=False)
    return sample[:, idx, :, :]

def PadSequence(**kwargs):
    sample = kwargs['sample']
    size = kwargs['size']

    C, T, V, M = sample.shape

    if T < size:
        padedSample = np.zeros((C, size, V, M))
        padedSample[:, :T, :, :] = sample
        return padedSample
    
    return sample

'''
angle_choices=[-10., -5., 0, 5., 10.],
scale_choices=[0.9, 1.0, 1.1],
translate_choices=[-0.2, -0.1, 0.0, 0.1, 0.2],
tsmoothness_choices=[1]
'''
def RandomAffineTransformAcrossTime(**kwargs):
    sample = kwargs['sample']
    angle_choices = kwargs['angle_choices']
    scale_choices = kwargs['scale_choices']
    translate_choices = kwargs['translate_choices']
    tsmoothness_choices = kwargs['tsmoothness_choices']

    C, T, V, M = sample.shape

    stepInTime = np.random.choice(tsmoothness_choices, 1)
    timeInstantsOfChange = np.arange(0, T, T * 1.0 / stepInTime).round().astype(int)
    timeInstantsOfChange = np.append(timeInstantsOfChange, T)
    numChanges = len(timeInstantsOfChange)

    chosenAngles = np.random.choice(angle_choices, numChanges)
    chosenScales = np.random.choice(scale_choices, numChanges)
    chosenTranslateX = np.random.choice(translate_choices, numChanges)
    chosenTranslateY = np.random.choice(translate_choices, numChanges)

    anglePerTimeInstant = np.zeros(T)
    scalePerTimeInstant = np.zeros(T)
    xTranslatePerTimeInstant = np.zeros(T)
    yTranslatePerTimeInstant = np.zeros(T)

    for i in range(numChanges - 1):
        anglePerTimeInstant[timeInstantsOfChange[i] : timeInstantsOfChange[i+1]] = np.linspace(
            chosenAngles[i], chosenAngles[i+1],
            timeInstantsOfChange[i+1] - timeInstantsOfChange[i]) * np.pi / 180

        scalePerTimeInstant[timeInstantsOfChange[i] : timeInstantsOfChange[i+1]] = np.linspace(
            chosenScales[i], chosenScales[i+1],
            timeInstantsOfChange[i+1] - timeInstantsOfChange[i])

        xTranslatePerTimeInstant[timeInstantsOfChange[i] : timeInstantsOfChange[i+1]] = np.linspace(
            chosenTranslateX[i], chosenTranslateX[i+1],
            timeInstantsOfChange[i+1] - timeInstantsOfChange[i])

        yTranslatePerTimeInstant[timeInstantsOfChange[i] : timeInstantsOfChange[i+1]] = np.linspace(
            chosenTranslateY[i], chosenTranslateY[i+1],
            timeInstantsOfChange[i+1] - timeInstantsOfChange[i])

    theta = np.array([[np.cos(anglePerTimeInstant) * scalePerTimeInstant,
                        -np.sin(anglePerTimeInstant) * scalePerTimeInstant],
                      [np.sin(anglePerTimeInstant) * scalePerTimeInstant,
                        np.cos(anglePerTimeInstant) * scalePerTimeInstant]])

    # Carry out the affine transformations
    for idx in range(T):
        xy = sample[0:2, idx, :, :]
        new_xy = np.dot(theta[:, :, idx], xy.reshape(2, -1))
        new_xy[0] += xTranslatePerTimeInstant[idx]
        new_xy[1] += yTranslatePerTimeInstant[idx]
        sample[0:2, idx, :, :] = new_xy.reshape(2, V, M)

    return sample

def RandomTemporalShift(**kwargs):
    sample = kwargs['sample']
    T = sample.shape[1]

    shiftedSample = np.zeros_like(sample)
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()

    size = end - start
    randomStart = np.random.randint(0, T - size)
    shiftedSample[:, randomStart:randomStart+size, :, :] = sample[:, start:end, :, :]

    return shiftedSample

'''
mean_choices=[0., -0.1, 0.1]
sigma_choices=[0.1, 0.2]
noise_level_choices=[0.002, 0.004, 0.005]
'''
def RandomGaussianNoise(**kwargs):
    sample = kwargs['sample']
    noise_level_choices = kwargs['noise_level_choices']
    mean_choices = kwargs['mean_choices']
    sigma_choices = kwargs['sigma_choices']

    noisy_sample = np.zeros_like(sample)
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()

    sample = sample[:, start:end, :, :]
    t = sample.shape[1]
    mean = np.random.choice(mean_choices, 1)
    sigma = np.random.choice(sigma_choices, 1)
    noise_level = np.random.choice(noise_level_choices, 1)

    noise = np.random.normal(mean, sigma, 1)

    nsample = sample + (noise * noise_level)
    noisy_sample[:, start:end, :, :] = nsample

    return noisy_sample