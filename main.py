import tensorflow as tf
from VDSR_MODEL import VDSR
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epoch',80,"Number of Epoch")
flags.DEFINE_integer('image_size',41,"The size of image input")
flags.DEFINE_integer('label_size',41,"The size of image output")
flags.DEFINE_boolean('is_train',True,"train or test?")
flags.DEFINE_string('test_img',"./data/test",'The name of test_img')
flags.DEFINE_string('model_path',"",'restore model...')
flags.DEFINE_float('clip_gradient',1e-1,"The clip gradent")
flags.DEFINE_integer("layer",20,"the size of layer")
flags.DEFINE_integer('batch_size',64,'The size of batch')
flags.DEFINE_float('learning_rate',1e-4,"The learning_rate set initial 0.01%")
flags.DEFINE_float('learning_rate_decay_rate',0.1,'The rate of learning_rate_decay')
flags.DEFINE_float('learning_rate_step_size',20,'learning_rate_decay step size')
flags.DEFINE_string('test_data_path', './data/test','The path of test data')
flags.DEFINE_string('train_data_path', './data/train','The path of train data')

def main(_) :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        vdsr = VDSR(sess, flags=FLAGS)
        if FLAGS.is_train:
            vdsr.train()
        else:
            vdsr.test()

if __name__ =='__main__' :
    tf.app.run()


