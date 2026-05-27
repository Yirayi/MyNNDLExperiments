import preProcessing
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,BatchNormalization,Activation
from keras.optimizers import Adam
import numpy as np

from tensorflow.keras.callbacks import ReduceLROnPlateau
cnn = Sequential()

# ── 第一组卷积块 ──────────────────────────
cnn.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))     # ← BN后再激活

cnn.add(Conv2D(32, (3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))           # ← 改为 0.2

# ── 第二组卷积块 ──────────────────────────
cnn.add(Conv2D(64, (3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(Conv2D(64, (3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.3))

# ── 第三组卷积块───────────────────────────
cnn.add(Conv2D(128, (3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(Conv2D(128, (3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.4))

# ── 全连接层 ──────────────────────────────
cnn.add(Flatten())
# 此时形状：4×4×128 = 2048

cnn.add(Dense(256, activation='relu'))  # ← 从512降到256
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(10, activation='softmax'))

cnn.summary()
# ── 编译 ──────────────────────────────────
cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 学习率衰减 ────────────────────────────
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # 学习率乘以0.5
    patience=5,       # 5轮没改善就衰减
    min_lr=1e-6
)
import tensorflow as tf

def augment_fn(image, label):
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)
    # 随机亮度
    image = tf.image.random_brightness(image, max_delta=0.1)
    # 随机对比度
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # 随机裁剪（先padding再裁回原始大小）
    image = tf.pad(image, [[4,4],[4,4],[0,0]])  # 四周补4像素
    image = tf.image.random_crop(image, size=[32,32,3])
    return image, label

# ── 数据增强 ──────────────────────────────
train_data, test_data, train_label, test_label = preProcessing.Get_Data()
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
dataset = (dataset
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)  # 并行增强
    .shuffle(1000)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)   # ← GPU训练时，CPU提前准备下一批
)

# ── 训练 ──────────────────────────────────
history_cnn=cnn.fit(
    dataset,                                    # 直接传入，不需要单独传 label
    epochs=50,
    validation_data=(test_data,test_label),                # 验证集同理
    callbacks=[lr_scheduler]
)
print(history_cnn.history.keys())#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(np.array(history_cnn.history['loss']))
plt.plot(np.array(history_cnn.history['val_loss']))
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.legend(['loss', 'val_loss'])
plt.show()

plt.figure(2)
plt.plot(np.array(history_cnn.history['accuracy']))
plt.plot(np.array(history_cnn.history['val_accuracy']))
plt.xlabel('Epoch')
plt.ylabel('Train acc')
plt.legend(['acc', 'val_acc'])
plt.show()

plt.figure(3)
plt.plot(np.array(history_cnn.history['lr']))
plt.xlabel('Epoch')
plt.ylabel('LearningRate')
plt.legend(['LR'])
plt.show()

cnn.save('model/cnnfirst.h5')