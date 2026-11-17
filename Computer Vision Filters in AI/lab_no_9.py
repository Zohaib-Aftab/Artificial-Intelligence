"""Task No 1
Classical Filters for Noisy Document Cleanup
A hospital scans paper reports, but the scans contain "salt-and-pepper" speckle noise and slightly blurred text. Before the scans are passed to an AI text-recognition system, the images must be cleaned and the text edges made clearer. Use classical filters to design the cleanup step.
Task Statement :Use the sample image skimage.data.page() (or any scanned text image you choose)"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
img = data.page()
print(f"Image Shape: {img.shape} | Data Type: {img.dtype} | Pixel Range: [{img.min()}, {img.max()}]")
np.random.seed(42)
noisy = img.copy()
total_pixels = noisy.size
num_noisy = int(0.05 * total_pixels)
random_indices = np.random.choice(total_pixels, num_noisy, replace=False)
half = num_noisy // 2
noisy.flat[random_indices[:half]] = 0
noisy.flat[random_indices[half:]] = 255
mean_blur = cv2.blur(noisy, (5, 5))
gauss_blur = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)
median_blur = cv2.medianBlur(noisy, 5)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
denoising_results = [
    ("Noisy (5% S&P)", noisy),
    ("Mean 5x5", mean_blur),
    ("Gaussian 5x5", gauss_blur),
    ("Median 5x5", median_blur)
]
for ax, (title, im) in zip(axes, denoising_results):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
psnr_mean = cv2.PSNR(img, mean_blur)
psnr_gauss = cv2.PSNR(img, gauss_blur)
psnr_median = cv2.PSNR(img, median_blur)
print(f"PSNR (Clean vs Mean 5x5):     {psnr_mean:.2f} dB")
print(f"PSNR (Clean vs Gaussian 5x5): {psnr_gauss:.2f} dB")
print(f"PSNR (Clean vs Median 5x5):   {psnr_median:.2f} dB")
sharpen_k = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
sharpened = cv2.filter2D(median_blur, -1, sharpen_k)
sobel_x = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(np.hypot(sobel_x, sobel_y))
canny_pair1 = cv2.Canny(sharpened, 50, 150)
canny_pair2 = cv2.Canny(sharpened, 100, 200)
diag_k = np.array([
    [2,  1,  0],
    [1,  0, -1],
    [0, -1, -2]
], dtype=np.float32)
diag_edges = cv2.filter2D(sharpened, -1, diag_k)
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
edge_results = [
    ("Sharpened (Median)", sharpened),
    ("Sobel Magnitude", sobel),
    ("Canny (50, 150)", canny_pair1),
    ("Canny (100, 200)", canny_pair2),
    ("Custom Diagonal 3x3", diag_edges)
]
for ax, (title, im) in zip(axes, edge_results):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
def my_conv2d(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32), ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out
patch = sharpened[:32, :32]
manual_out = my_conv2d(patch, sharpen_k)
opencv_out = cv2.filter2D(patch.astype(np.float32), -1, sharpen_k, borderType=cv2.BORDER_CONSTANT)
print("Max abs difference (my_conv2d vs cv2.filter2D):", np.abs(manual_out - opencv_out).max())

"""Task No 2
Use the Fashion-MNIST dataset available in TensorFlow/Keras (10 classes of 28x28 grayscale clothing images) together with the pretrained ResNet-50 model.
Task Statement
Compare a CNN trained from scratch against a pretrained ResNet-50 used as a filter bank."""
import numpy as np, time, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
train_X = train_X.astype("float32") / 255.0
test_X  = test_X.astype("float32") / 255.0
print("Train X:", train_X.shape, "| Train y:", train_y.shape)
print("Test  X:", test_X.shape,  "| Test  y:", test_y.shape)
print("Number of classes:", len(np.unique(train_y)))
train_Xc = train_X[..., None]
test_Xc  = test_X[..., None]
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.times = []
    def on_epoch_begin(self, epoch, logs=None): self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None): self.times.append(time.time() - self.t0)
cnn = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.summary()
cnn_time = TimeHistory()
cnn.fit(train_Xc, train_y, epochs=5, batch_size=64, validation_split=0.1, callbacks=[cnn_time])
cnn_loss, cnn_acc = cnn.evaluate(test_Xc, test_y, verbose=0)
print(f"\nCNN  Test loss: {cnn_loss:.4f} | Test accuracy: {cnn_acc:.4f}")
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
base = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
def to_resnet_input(x):
    x = tf.expand_dims(x, -1)
    x = tf.repeat(x, 3, axis=-1)
    x = tf.image.resize(x, (64, 64))
    return preprocess_input(x * 255.0)
N_TRAIN = 10000
tr_X, tr_y = train_X[:N_TRAIN], train_y[:N_TRAIN]
train_ds = (tf.data.Dataset.from_tensor_slices((tr_X, tr_y))
            .map(lambda x, y: (to_resnet_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(2000).batch(64).prefetch(tf.data.AUTOTUNE))
test_ds  = (tf.data.Dataset.from_tensor_slices((test_X, test_y))
            .map(lambda x, y: (to_resnet_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(64).prefetch(tf.data.AUTOTUNE))
for xb, yb in train_ds.take(1):
    print("ResNet input batch:", xb.shape)
    base.trainable = False
resnet_model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
resnet_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
res_time = TimeHistory()
resnet_model.fit(train_ds, epochs=3, callbacks=[res_time])
res_loss, res_acc = resnet_model.evaluate(test_ds, verbose=0)
print(f"\nResNet-50 (frozen) Test loss: {res_loss:.4f} | Test accuracy: {res_acc:.4f}")
w, b = base.get_layer("conv1_conv").get_weights()
print("conv1 filter shape:", w.shape)
w_show = (w - w.min()) / (w.max() - w.min())
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(w_show[:, :, :, i]); ax.axis("off")
plt.suptitle("ResNet-50 conv1 filters (64)"); plt.show()
img_idx = 0
sample = to_resnet_input(test_X[img_idx])[None, ...]
fm_layers = ["conv1_relu", "conv3_block4_out"]
extractor = tf.keras.Model(base.input, [base.get_layer(n).output for n in fm_layers])
maps = extractor.predict(sample, verbose=0)
plt.figure(figsize=(3, 3)); plt.imshow(test_X[img_idx], cmap="gray")
plt.title(class_names[test_y[img_idx]]); plt.axis("off"); plt.show()
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for r, (name, fm) in enumerate(zip(fm_layers, maps)):
    print(name, fm.shape)
    for c in range(8):
        axes[r, c].imshow(fm[0, :, :, c], cmap="viridis"); axes[r, c].axis("off")
    axes[r, 0].set_title(name, fontsize=9, loc="left")
plt.tight_layout(); plt.show()
import pandas as pd
def trainable_params(m):
    return int(np.sum([np.prod(v.shape) for v in m.trainable_weights]))
table = pd.DataFrame({
    "Model": ["CNN from scratch", "ResNet-50 (frozen base + head)"],
    "Trainable parameters": [trainable_params(cnn), trainable_params(resnet_model)],
    "Training time / epoch (s)": [round(np.mean(cnn_time.times), 1), round(np.mean(res_time.times), 1)],
    "Epochs": [5, 3],
    "Training images": [60000, 10000],
    "Test accuracy": [round(cnn_acc, 4), round(res_acc, 4)],
    "Test loss": [round(cnn_loss, 4), round(res_loss, 4)],
})

"""Task No 3
Use any 2 photographs of your choice (for example a flower and a vehicle) and the pretrained ResNet-50 model from Part B2. No training is required.
Task Statement
Study the filters of a pretrained ResNet-50 and compare them with classical filters."""
import numpy as np, cv2, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
model = ResNet50(weights="imagenet")
print("Number of layers :", len(model.layers))
print("Total parameters :", f"{model.count_params():,}")
from skimage import data
USE_UPLOAD = False
if USE_UPLOAD:
    from google.colab import files
    up = files.upload()
    names = list(up.keys())[:2]
    raw_imgs = [cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2RGB) for n in names]
    titles = names
else:
    raw_imgs = [data.chelsea(), data.rocket()]
    titles = ["Cat (chelsea)", "Rocket"]
imgs224 = [cv2.resize(im, (224, 224)) for im in raw_imgs]
xs = [preprocess_input(np.expand_dims(im.astype("float32"), 0)) for im in imgs224]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, im, t in zip(axes, imgs224, titles):
    ax.imshow(im); ax.set_title(t); ax.axis("off")
plt.show()
for t, x in zip(titles, xs):
    preds = model.predict(x, verbose=0)
    print(f"\nTop-3 for {t}:")
    for _, name, prob in decode_predictions(preds, top=3)[0]:
        print(f"   {name:25s} {prob:.3f}")
        w, b = model.get_layer("conv1_conv").get_weights()
print("conv1 filter shape:", w.shape)       # (7, 7, 3, 64)
w_show = (w - w.min()) / (w.max() - w.min())
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(w_show[:, :, :, i]); ax.set_title(str(i), fontsize=7); ax.axis("off")
plt.suptitle("ResNet-50 conv1 filters (index shown above each filter)"); plt.tight_layout(); plt.show()
layer_names = ["conv1_relu", "conv3_block4_out", "conv5_block3_out"]
extractor = Model(model.input, [model.get_layer(n).output for n in layer_names])
all_maps = []
for t, x in zip(titles, xs):
    maps = extractor.predict(x, verbose=0)
    all_maps.append(maps)
    print(f"\n{t}")
    for n, fm in zip(layer_names, maps):
        print(f"   {n:20s} output shape: {fm.shape}")
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
    for r, (n, fm) in enumerate(zip(layer_names, maps)):
        for c in range(8):
            axes[r, c].imshow(fm[0, :, :, c], cmap="viridis"); axes[r, c].axis("off")
        axes[r, 0].set_title(f"{n} {fm.shape[1:]}", fontsize=9, loc="left")
    plt.suptitle(t); plt.tight_layout(); plt.show()
for t, im, maps in zip(titles, imgs224, all_maps):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    fm = maps[0][0]
    mag_s = cv2.resize(mag, (112, 112))
    corr = [np.corrcoef(mag_s.ravel(), fm[:, :, k].ravel())[0, 1] if fm[:, :, k].std() > 0 else 0
            for k in range(64)]
    top2 = np.argsort(np.nan_to_num(corr))[::-1][:2]
    print(f"{t}: conv1 maps most similar to Sobel magnitude -> {top2.tolist()}, "
          f"correlation = {[round(float(corr[k]), 2) for k in top2]}")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(gray, cmap="gray");           axes[0].set_title("Grayscale")
    axes[1].imshow(np.abs(sx), cmap="gray");     axes[1].set_title("Sobel-X")
    axes[2].imshow(np.abs(sy), cmap="gray");     axes[2].set_title("Sobel-Y")
    axes[3].imshow(fm[:, :, top2[0]], cmap="gray"); axes[3].set_title(f"conv1 map {top2[0]}")
    axes[4].imshow(fm[:, :, top2[1]], cmap="gray"); axes[4].set_title(f"conv1 map {top2[1]}")
    for a in axes: a.axis("off")
    plt.suptitle(t); plt.tight_layout(); plt.show()