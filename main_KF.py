import time
from filter_v8 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '128', """batch size""")
tf.app.flags.DEFINE_integer('max_seq', '100', """number of measurements included""")
tf.app.flags.DEFINE_integer('max_seq_len', '5000', """number of measurements included""")
tf.app.flags.DEFINE_integer('eval_interval', '25', """Batch interval for Evaluating Model""")
tf.app.flags.DEFINE_integer('plot_interval', '10', """Batch interval for Plotting Model Training Results""")
tf.app.flags.DEFINE_integer('checkpoint_interval', '10', """Batch interval for Saving Model""")
tf.app.flags.DEFINE_integer('num_state', '12', """State Dimension""")
tf.app.flags.DEFINE_integer('num_meas', '3', """Measurement Dimension""")
tf.app.flags.DEFINE_integer('num_mixtures', '4', """Measurement Dimension""")
tf.app.flags.DEFINE_integer('F_hidden', '18', """RNN Hidden Units""")
tf.app.flags.DEFINE_integer('default_data_rate', '25', """Default Data Rate in HZ""")

tf.app.flags.DEFINE_float('max_epoch', 100, """Maximum Dataset Epochs""")
tf.app.flags.DEFINE_float('dropout_rate', 1.0, """Dropout Probability during training""")

tf.app.flags.DEFINE_float('learning_rate', '1e-3', """Default Data Rate in HZ""")

training = True


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        data_dir = 'D:/TrackFilterData/OOPBroad'
        filter_name = 'filter_v8'
        save_dir = './' + filter_name
        ckpt_dir = save_dir + '/checkpoints/'
        log_dir = save_dir + '/logs/'

        plt_dir = save_dir + '/plots/'
        plot_eval_dir = save_dir + '/plots_eval/'
        plot_test_dir = save_dir + '/plots_test/'

        check_folder(save_dir)
        check_folder(ckpt_dir)
        check_folder(log_dir)
        check_folder(plt_dir)
        check_folder(plot_eval_dir)
        check_folder(plot_test_dir)

        if training is True:
            filter_train = Filter(sess, state_type='INDYGRU', mode='training',
                                  data_dir=data_dir, filter_name=filter_name, save_dir=save_dir,
                                  F_hidden=FLAGS.F_hidden, num_state=FLAGS.num_state, num_meas=FLAGS.num_meas,
                                  max_seq=FLAGS.max_seq, num_mixtures=FLAGS.num_mixtures, batch_size=FLAGS.batch_size,
                                  learning_rate=FLAGS.learning_rate, plot_interval=FLAGS.plot_interval,
                                  checkpoint_interval=FLAGS.checkpoint_interval, dropout_rate=FLAGS.dropout_rate,
                                  constant=False, decimate_data=True)

            print('Building filter')
            t0 = time.time()

            filter_train.build_model()

            t01 = time.time()
            dti = t01 - t0
            print('Filter Completed :: ' + str(dti) + ' seconds to complete.')

            filter_train.train(data_rate=FLAGS.default_data_rate, max_exp_seq=FLAGS.max_seq_len)

            print(" [*] Training finished!")

        else:
            filter_test = Filter(sess, state_type='INDYGRU', mode='testing',
                                 data_dir=data_dir, filter_name=filter_name, save_dir=save_dir,
                                 F_hidden=FLAGS.F_hidden, num_state=FLAGS.num_state, num_meas=FLAGS.num_meas,
                                 max_seq=1, num_mixtures=FLAGS.num_mixtures, batch_size=FLAGS.batch_size,
                                 learning_rate=FLAGS.learning_rate, plot_interval=FLAGS.plot_interval,
                                 checkpoint_interval=FLAGS.checkpoint_interval, dropout_rate=FLAGS.dropout_rate,
                                 constant=False, decimate_data=False)

            print('Building filter')
            t0 = time.time()

            filter_test.build_model()

            t01 = time.time()
            dti = t01 - t0
            print('Filter Completed :: ' + str(dti) + ' seconds to complete.')

            filter_test.test(data_rate=FLAGS.default_data_rate, max_exp_seq=FLAGS.max_seq_len)

            print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
