import time
from filter_d1 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '2', """batch size""")
tf.app.flags.DEFINE_integer('max_seq', '50', """number of measurements included""")
tf.app.flags.DEFINE_integer('max_seq_len', '5000', """number of measurements included""")
tf.app.flags.DEFINE_integer('plot_interval', '1', """Batch interval for Plotting Model Training Results""")
tf.app.flags.DEFINE_integer('default_data_rate', '25', """Default Data Rate in HZ""")

tf.app.flags.DEFINE_float('dropout_rate', 1.0, """Dropout Probability during training""")


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        data_dir = 'D:/TrackFilterData/OOPBroad'
        filter_name = 'filter_d1'
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

        filter_test = Filter(sess, mode='testing', data_dir=data_dir, filter_name=filter_name, save_dir=save_dir,
                             max_seq=1, batch_size=FLAGS.batch_size, plot_interval=FLAGS.plot_interval)

        print('Building filter')
        t0 = time.time()

        filter_test.build_model()

        t01 = time.time()
        dti = t01 - t0
        print('Filter Completed :: ' + str(dti) + ' seconds to complete.')

        filter_test.test(x_data=x_data, ecef_ref=ecef_ref, lla_data=lla_data, data_rate=FLAGS.default_data_rate, max_exp_seq=FLAGS.max_seq_len)

        print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
