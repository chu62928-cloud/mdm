import numpy as np

path = "/root/autodl-tmp/motion-diffusion-model/save/263_batch_single_outputs/0001_a_person_walks_forward/results.npy"
obj = np.load(path, allow_pickle=True)

print("loaded type:", type(obj))
print("shape:", getattr(obj, "shape", None))
print("dtype:", getattr(obj, "dtype", None))

if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
    data = obj.item()
    print("\nDetected dict-like results.npy")
    print("keys:", list(data.keys()))

    motion = np.asarray(data["motion"])
    lengths = np.asarray(data["lengths"])
    text = data["text"]

    print("motion.shape =", motion.shape)
    print("lengths.shape =", lengths.shape)
    print("num text =", len(text))
    print("num_samples =", data.get("num_samples"))
    print("num_repetitions =", data.get("num_repetitions"))

    sample0 = motion[0]
    print("\nsingle sample raw shape =", sample0.shape)

    if sample0.ndim == 3:
        print("If this is official MDM format, expected raw single-sample shape is likely (J, 3, T).")
        print("Converted shape should be:", np.transpose(sample0, (2, 0, 1)).shape, "=(T, J, 3)")
else:
    arr = np.asarray(obj)
    print("\nDetected plain ndarray")
    print("array shape =", arr.shape)