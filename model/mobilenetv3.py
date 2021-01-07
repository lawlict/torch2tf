import tensorflow as tf 
from tensorflow.keras import layers


class ASP(tf.keras.layers.Layer):
    def __init__(self, in_size, encoder_size):
        super().__init__()
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=encoder_size, kernel_size=1, activation='tanh'),
            tf.keras.layers.Conv1D(filters=in_size, kernel_size=1),
            tf.keras.layers.Softmax(axis=1),
        ])

    def call(self, x):
        w = self.attention(x)
        mu = tf.reduce_sum(x * w, axis=1)
        sg = tf.sqrt(tf.clip_by_value(tf.reduce_sum((x**2)*w, axis=1)-mu**2, 1e-5, tf.float32.max))
        x = tf.concat((mu, sg), axis=1)
        return x


class hswish(tf.keras.layers.Layer):
    def call(self, x):
        out = x * tf.nn.relu6(x + 3) / 6
        return out


class hsigmoid(tf.keras.layers.Layer):
    def call(self, x):
        out = tf.nn.relu6(x + 3) / 6
        return out


class AdaptiveAvgPool2d(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.backend.mean(x, axis=[1, 2], keepdims=True)


class SeModule(tf.keras.layers.Layer):
    def __init__(self, in_size, reduction=4):
        super().__init__()
        self.se = tf.keras.Sequential([
            AdaptiveAvgPool2d(),
            tf.keras.layers.Conv2D(filters=in_size//reduction, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=in_size, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            hsigmoid()
        ])

    def call(self, x):
        return x * self.se(x)


class Block(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, strides):
        super().__init__()
        self.se = semodule

        self.conv1 = tf.keras.layers.Conv2D(filters=expand_size, kernel_size=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.nolinear1 = nolinear
        
        self.pad = tf.keras.layers.ZeroPadding2D(kernel_size // 2)
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.nolinear2 = nolinear

        self.conv3 = tf.keras.layers.Conv2D(filters=out_size, kernel_size=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-5)

        self.shortcut = tf.keras.Sequential()
        if strides != 1 or in_size != out_size:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=out_size, kernel_size=1, strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization(epsilon=1e-5)
            ])

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nolinear1(out)
        out = self.nolinear2(self.bn2(self.conv2(self.pad(out))))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x)
        return out
    

class MobileNetV3_Small(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.pad = tf.keras.layers.ZeroPadding2D(1)
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.hs1 = hswish()

        self.bneck = tf.keras.Sequential([
            Block(5, 16, 16, 16, tf.keras.layers.ReLU(), SeModule(16), 2),
            Block(5, 16, 72, 24, tf.keras.layers.ReLU(), None, 2),
            Block(3, 24, 88, 24, tf.keras.layers.ReLU(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        ])

        self.conv2 = tf.keras.layers.Conv2D(filters=576, kernel_size=1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.hs2 = hswish()

        self.pooling = ASP(576, 128)
        self.linear4 = tf.keras.layers.Dense(128)

    def call(self, x):
        x = tf.expand_dims(x, axis=3)
        out = self.hs1(self.bn1(self.conv1(self.pad(x))))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))

        out = tf.reduce_mean(out, axis=2)
        out = self.pooling(out)
        out = self.linear4(out)
        return out


if __name__ == '__main__':
    model = MobileNetV3_Small()
    model.build(input_shape=(1, 150, 40))
    model.summary()

    x = np.ones([1, 150, 40], dtype=np.float32)
    y = model(x)
    model.save_weights('bin/tf_weights.h5')
