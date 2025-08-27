import json, os, numpy as np, tensorflow as tf
from tensorflow.keras import layers

# ---- Paths ----
TRAIN_DIR  = "id_card_app/dataset/Train"
TEST_DIR   = "id_card_app/dataset/Test"
MODEL_PATH = "model.h5"
META_PATH  = "metadata.json"

# ---- Config ----
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 123
UNFREEZE_N = 50
HEAD_EPOCHS     = 12
FINETUNE_EPOCHS = 12
EARLYSTOP_PATIENCE = 4

# ---- Datasets  ----
data_aug = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.1),
], name="data_augmentation")

raw_train = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset='training', seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
raw_val = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset='validation', seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# Class names
class_names = getattr(raw_train, "class_names", None)
if not class_names:
    class_names = sorted([d for d in os.listdir(TRAIN_DIR)
                          if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
if num_classes < 2:
    raise ValueError(f"Need at least 2 classes. Found: {num_classes} ({class_names})")

print("Classes:", class_names)

# Cache + prefetch
train_ds = raw_train.cache().prefetch(tf.data.AUTOTUNE)
val_ds   = raw_val.cache().prefetch(tf.data.AUTOTUNE)

# Optional test
test_ds = None
if os.path.exists(TEST_DIR):
    try:
        raw_test = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
        )
        test_ds = raw_test.cache().prefetch(tf.data.AUTOTUNE)
    except Exception:
        pass

# ---- Model ----
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet'
)
base.trainable = False

head = [
    layers.Input(shape=(*IMG_SIZE,3)),
    data_aug,
    layers.Rescaling(1./255),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
]

if num_classes == 2:
    head.append(layers.Dense(1, activation='sigmoid'))
    loss = 'binary_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    mode = 'binary'
else:
    head.append(layers.Dense(num_classes, activation='softmax'))
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']  # you can add tf.keras.metrics.AUC(multi_label=False) if desired
    mode = 'multiclass'

model = tf.keras.Sequential(head, name="id_model")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy' if mode=='multiclass' else 'val_auc',
    mode='max' if mode=='multiclass' else 'max',
    patience=EARLYSTOP_PATIENCE, restore_best_weights=True
)

print("\n=== Phase 1: Train head ===")
_ = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=[early], verbose=1)

# Fine-tune top of backbone
print("\n=== Phase 2: Fine-tune backbone top ===")
base.trainable = True
for layer in base.layers[:-UNFREEZE_N]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss, metrics=metrics)

early_ft = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy' if mode=='multiclass' else 'val_auc',
    mode='max', patience=EARLYSTOP_PATIENCE, restore_best_weights=True
)

_ = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=[early_ft], verbose=1)

# ---- Save metadata ----
meta = {
    "class_names": class_names,
    "img_size": list(IMG_SIZE),
    "mode": mode
}

if mode == 'binary':
    # find best threshold on validation
    y_true, y_prob = [], []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0).ravel()
        y_true.append(yb.numpy()); y_prob.append(p)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)

    def f1_at_t(y_true, y_prob, t):
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred==1)&(y_true==1)); fp = np.sum((pred==1)&(y_true==0)); fn = np.sum((pred==0)&(y_true==1))
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        return 2*prec*rec/(prec+rec+1e-9), prec, rec

    thresholds = np.linspace(0.05,0.95,19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1,_,_ = f1_at_t(y_true,y_prob,t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    meta["positive_class_index"] = 1
    meta["positive_class_name"]  = class_names[1]
    meta["threshold"] = float(best_t)
else:
    # softmax: we’ll use a simple reject threshold for "uncertain"
    meta["reject_threshold"] = 0.60  # you can tune this later if needed

model.save(MODEL_PATH)
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {MODEL_PATH} and {META_PATH}\nMode: {mode}  Classes: {class_names}")
