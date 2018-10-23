import time
from KF_vaescan16 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '32', """batch size""")
tf.app.flags.DEFINE_integer('max_seq', '100', """number of measurements included""")
tf.app.flags.DEFINE_integer('max_seq_len', '2750', """number of measurements included""")
tf.app.flags.DEFINE_integer('eval_interval', '10', """number of measurements included""")
tf.app.flags.DEFINE_integer('future_seq', '1', """Number of Steps to Predict""")
tf.app.flags.DEFINE_integer('num_state', '12', """State Dimension""")
tf.app.flags.DEFINE_integer('num_meas', '3', """Measurement Dimension""")
tf.app.flags.DEFINE_integer('num_mixtures', '5', """Measurement Dimension""")
tf.app.flags.DEFINE_float('max_epoch', 100, """Maximum Dataset Epochs""")
tf.app.flags.DEFINE_float('RE', 6378137, """Radius of Earth""")
tf.app.flags.DEFINE_float('GM', 398600441890000, """GM""")
tf.app.flags.DEFINE_integer('F_hidden', '8', """RNN Hidden Units""")
tf.app.flags.DEFINE_integer('default_data_rate', '25', """Default Data Rate in HZ""")


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        data_dir = 'D:/TrackFilterData/Delivery_1/AdvancedBroad'
        filter_name = 'KF_vae16'
        save_dir = './' + filter_name
        ckpt_dir = save_dir + '/checkpoints/'
        log_dir = save_dir + '/logs/'

        plt_dir = save_dir + '/plots/'
        plot_eval_dir = save_dir + '/plots_eval/'
        plot_test_dir = save_dir + '/plots_test/'

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        if not os.path.isdir(plt_dir):
            os.mkdir(plt_dir)

        if not os.path.isdir(plot_eval_dir):
            os.mkdir(plot_eval_dir)

        if not os.path.isdir(plot_test_dir):
            os.mkdir(plot_test_dir)

        training = True

        if training is True:
            filter_train = Filter(sess, trainable_state=False, state_type='PLSTM', mode='training',
                                  data_dir=data_dir, filter_name=filter_name, save_dir=save_dir,
                                  F_hidden=FLAGS.F_hidden, num_state=FLAGS.num_state, num_meas=FLAGS.num_meas,
                                  max_seq=FLAGS.max_seq, num_mixtures=FLAGS.num_mixtures, batch_size=FLAGS.batch_size, constant=True)

            print('Building filter')
            t0 = time.time()

            filter_train.build_model(is_training=training)

            t01 = time.time()
            dti = t01 - t0
            print('Filter Completed :: ' + str(dti) + ' seconds to complete.')

            filter_train.train(data_rate=FLAGS.default_data_rate, max_exp_seq=FLAGS.max_seq_len)

            print(" [*] Training finished!")
        else:
            filter_test = Filter(sess, trainable_state=False, state_type='PLSTM', mode='testing',
                                 data_dir=data_dir, filter_name=filter_name, save_dir=save_dir,
                                 F_hidden=FLAGS.F_hidden, num_state=FLAGS.num_state, num_meas=FLAGS.num_meas,
                                 max_seq=1, num_mixtures=FLAGS.num_mixtures, batch_size=FLAGS.batch_size, constant=False)

            print('Building filter')
            t0 = time.time()

            filter_test.build_model(is_training=training)

            t01 = time.time()
            dti = t01 - t0
            print('Filter Completed :: ' + str(dti) + ' seconds to complete.')

            filter_test.test(data_rate=FLAGS.default_data_rate, max_exp_seq=FLAGS.max_seq_len)

            print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
