import tensorflow as tf

def conv_layer(x, layers, filter_out, ks, stride, rel=True, train=False):
    enc = tf.keras.layers.Conv2D(filter_out, ks, strides=stride, padding='SAME')
    layers['conv_'+str(len(list(layers)))] = enc
    l = enc(x)
    
    if train:
        enc = tf.keras.layers.BatchNormalization()
        layers['batch_'+str(len(list(layers)))] = enc
        l = enc(l, training=True)
        
    if rel:
        enc = tf.keras.activations.relu#(enc)
        layers['rel_'+str(len(list(layers)))] = enc
        l = enc(l)
        
    return l

def conv_transpose_layer(x, layers, filter_out, ks, stride, train):
    dec = tf.keras.layers.Conv2DTranspose(filter_out, ks, strides=stride, padding='SAME')
    layers['conv_trans_'+str(len(list(layers)))] = dec
    l = dec(x)
    
    dec = tf.keras.layers.BatchNormalization()
    layers['batch_'+str(len(list(layers)))] = dec
    l = dec(l, training=True)
    
    dec = tf.keras.activations.relu
    layers['rel_'+str(len(list(layers)))] = dec
    l = dec(l)
    
    #dec = tf.keras.layers.Dropout(0.2)
    #layers['dropout_'+str(len(list(layers)))] = dec
    #l = dec(l, training=True)
    return l

def conv_down_block(x, layers, filter_out, ks, rel=True, train=False):
    lay = conv_layer(x, layers, filter_out=filter_out, ks=ks, stride=1, rel=rel, train=train)
    
    skip = conv_layer(lay, layers, filter_out=filter_out, ks=ks, stride=1, rel=rel, train=train)
    #print('first:', 'skip_'+str(len(list(layers))))
    layers['skip_'+str(len(list(layers)))] = 'skip'
    
    down = conv_layer(skip, layers, filter_out=filter_out, ks=ks, stride=2, rel=rel, train=train)
    return down, skip
    
def conv_up_block(x, skip, layers, filter_out, ks, train=False):
    up = conv_transpose_layer(x, layers, filter_out=filter_out, ks=ks, stride=2, train=train)
    lay = tf.concat((up,skip),3)
    
    #print('first:', 'concat_'+str(len(list(layers))))
    layers['concat_'+str(len(list(layers)))] = 'concat'
    
    lay = conv_layer(lay, layers, filter_out=filter_out, ks=ks, stride=1, train=train)
    lay = conv_layer(lay, layers, filter_out=filter_out, ks=ks, stride=1, train=train)
    return lay

def unet(imgs, train, num_outputs): #This unet is not good. Needs to go deeper and needs more convs between each layer.
    layers = {}

    norm = tf.keras.layers.BatchNormalization()#(imgs, training=train)# 256
    layers['batch_0'] = norm
    norm = norm(imgs, training=train)
    
    down_1, skip_1 = conv_down_block(norm, layers, filter_out=64, ks=3, train=train) # 256, 128 
    down_2, skip_2 = conv_down_block(down_1, layers, filter_out=64, ks=3, train=train) # 128, 64
    down_3, skip_3 = conv_down_block(down_2, layers, filter_out=64, ks=3, train=train) # 64, 32
    down_4, skip_4 = conv_down_block(down_3, layers, filter_out=128, ks=3, train=train) # 32, 16
    down_5, skip_5 = conv_down_block(down_4, layers, filter_out=128, ks=3, train=train) # 16, 8
    down_6, skip_6 = conv_down_block(down_5, layers, filter_out=256, ks=3, train=train) # 8, 4
    
    dense = conv_layer(down_6, layers, filter_out=256, ks=3, stride=1, train=train) 
    dense = conv_layer(dense, layers, filter_out=256, ks=3, stride=1, train=train)
    dense = conv_layer(dense, layers, filter_out=256, ks=3, stride=1, train=train)
    
    up_6 = conv_up_block(dense, skip_6, layers, filter_out=256, ks=3, train=train) # 4, 8
    up_5 = conv_up_block(up_6, skip_5, layers, filter_out=128, ks=3, train=train) # 8, 16
    up_4 = conv_up_block(up_5, skip_4, layers, filter_out=128, ks=3, train=train) # 16, 32
    up_3 = conv_up_block(up_4, skip_3, layers, filter_out=64, ks=3, train=train) # 32, 64
    up_2 = conv_up_block(up_3, skip_2, layers, filter_out=64, ks=3, train=train) # 64, 128
    up_1 = conv_up_block(up_2, skip_1, layers, filter_out=64, ks=3, train=train) # 128, 256
    out = conv_layer(up_1, layers, filter_out=num_outputs, ks=3, stride=1, rel=False, train=None)
    return out, layers

class Model_handler:
    def __init__(self, load_path=None):
        
        self.make_model()
        if load_path != None:
            self.model_true.load_weights(load_path)
    
    def call_net(self, inp, training=False):
        x = inp
        skip = []
        for lay in list(self.layers):
            #rint(lay)
            op = self.layers[lay]
            #print(op)
            if 'conv' in lay:
                x = op(x)
            elif 'conv_trans' in lay:
                x = op(x)
            elif 'batch' in lay:
                x = op(x, training=training)
            elif 'dropout' in lay:
                x = op(x, training=training)
            elif 'rel' in lay:
                x = op(x)
            elif 'concat' in lay:
                x = tf.concat((x,skip[-1]),3)
                skip = skip[:-1]
            elif 'skip' in lay:
                skip += [x]
            else:
                print('ERROR')
                print(lay)
                exit()
        if len(skip) > 0:
            print('ERROR')
            exit()
        return tf.nn.softmax(x, axis = -1, name='output')
    
    def make_model(self):
        
        self.inputs = tf.keras.Input(shape=(None, None, 3)) # channels 3, last time was 1 - BW did not work well, max was .67
        no_lay, self.layers = unet(self.inputs, True, num_outputs=2)
        
        self.softmax = self.call_net(self.inputs, training=True)
        no_lay = tf.nn.softmax(no_lay, axis = -1, name='output')
        
        self.model_true = tf.keras.Model(inputs=self.inputs, outputs=self.softmax)
        self.model_false = tf.keras.Model(inputs=self.inputs, outputs=self.call_net(self.inputs, training=False))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001) #was 0.001
        
        self.model_true.compile(
            optimizer=self.opt,
            loss=self.loss,
            metrics=[DiceScore(), CE()]
                          )
        
        self.model_false.compile(
            optimizer=self.opt,
            loss=self.loss,
            metrics=[DiceScore(), CE()]
                          )
        print('Done compile!')
    
    def dice_loss(self, y_true, y_pred):
        smoothing = 1.0
        one_hot = tf.one_hot(y_true, 2)
        num = tf.reduce_sum(2*y_pred*one_hot, [1, 2]) + smoothing
        den = tf.reduce_sum(  y_pred+one_hot, [1, 2]) + smoothing
        dice = num/den
        dice_avg = tf.reduce_mean(dice, name = 'dice_loss')
        return 1.0 - dice_avg#, dice

    def loss(self, y_true, y_pred):
        dice_loss = self.dice_loss(y_true, y_pred)
        ce_loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_true, 2), y_pred)
        return ce_loss + dice_loss
    
class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.bg = self.add_weight(name='bg', initializer='zeros')
        self.fg = self.add_weight(name='fg', initializer='zeros')
        
        self.arr_bg = []
        self.arr_fg = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        smoothing = 1.0
        one_hot = tf.one_hot(y_true, 2)
        num = tf.reduce_sum(2*y_pred*one_hot, [1, 2]) + smoothing
        den = tf.reduce_sum(  y_pred+one_hot, [1, 2]) + smoothing
        dice = num/den
        #self.avg.assign_add(dice_avg)
        dice_avg = tf.reduce_mean(dice,axis=0, name = 'dice_loss')
        self.arr_bg = dice[:,0]
        self.arr_fg = dice[:,1]
        
        self.bg.assign_add(dice_avg[0])
        self.fg.assign_add(dice_avg[1])

    def result(self):
        return {'d_bg':self.bg, 'd_fg':self.fg, 'arr_d_bg':self.arr_bg, 'arr_d_fg':self.arr_fg}
    
class CE(tf.keras.metrics.Metric):
    def __init__(self, name='ce', **kwargs):
        super(CE, self).__init__(name=name, **kwargs)
        self.ce = self.add_weight(name='ce', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #smoothing = 1.0
        one_hot = tf.one_hot(y_true, 2)
        log_sum = -tf.reduce_sum(tf.math.log(y_pred+0.000001)*one_hot, axis=-1)
        ce = tf.reduce_mean(log_sum)
        self.ce.assign_add(ce)

    def result(self):
        return {'ce':self.ce}