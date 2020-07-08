import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import time


from tf_model import SELDnet
#from mydatagenerator import MyDataGenerator
from dataregulator import DataRegulator
from augmentation import *

import parameter
import cls_feature_class
from metrics import evaluation_metrics, SELD_evaluation_metrics


# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_integer("task_id", 999, "evaluation or development?")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
tf.app.flags.DEFINE_integer("augment", 0, "Augmentation")

tf.app.flags.DEFINE_float("learning_rate", 0.0002, "Numer of training step to evaluate (default: 100)")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Numer of training step to evaluate (default: 100)")
tf.app.flags.DEFINE_integer("training_epoch", 2000, "Numer of training step to evaluate (default: 100)")

tf.app.flags.DEFINE_integer("evaluate_every", 100, "Numer of training step to evaluate (default: 100)")

tf.app.flags.DEFINE_integer("seq_len", 600, "Feature sequence length (default: 300)")

tf.app.flags.DEFINE_integer("early_stopping", 0, "Early stopping (default: 0)")
tf.app.flags.DEFINE_integer("patience", 50, "Number of evaluation without improvement to trigger early stopping (default: 50)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

print("augment")
print(FLAGS.augment)
print("learning_rate")
print(FLAGS.learning_rate)
print("decay rate")
print(FLAGS.decay_rate)
print("training_epoch")
print(FLAGS.training_epoch)

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

evaluate_every = FLAGS.evaluate_every
seq_len = FLAGS.seq_len

#learning schedule
scheduler = dict(
learning_rate = FLAGS.learning_rate,
decay_rate = FLAGS.decay_rate,
warmup_epoch = 10,
schedule = [200, 600, 1000, 9000, 9500],
training_epoch = FLAGS.training_epoch
)

params = parameter.get_params(str(FLAGS.task_id))
feat_cls = cls_feature_class.FeatureClass(params)

train_splits, train_check_splits, val_splits, test_splits = None, None, None, None

if params['mode'] == 'dev':
    test_splits = [1]
    val_splits = [2]
    train_splits = [3, 4, 5, 6]

elif params['mode'] == 'eval':
    test_splits = [7, 8]
    val_splits = []
    train_splits = [1, 2, 3, 4, 5, 6]

iseval = (params['mode'] == 'eval')

data_gen_train = DataRegulator(train_splits,
                               params['feat_label_dir'] + '{}_dev_label/'.format(params['dataset']),
                               params['feat_label_dir'] + '{}_dev_norm/'.format(params['dataset']),
                               seq_len=seq_len,
                               seq_hop=5) # hop len must be factor of 5

data_gen_train.load_data()
data_gen_train.shuffle_data()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement,
                                  gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = SELDnet(params, data_gen_train.get_in_shape(),
                      data_gen_train.get_out_shape_sed(),
                      data_gen_train.get_out_shape_doa())

        # change leaerning rate for Adam
        learning_rate = tf.placeholder(tf.float32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())

        def learning_schedule(epoch):
            divide_epoch = np.array(scheduler['schedule'])
            decay = sum(epoch >= divide_epoch)
            if epoch <= scheduler['warmup_epoch']:
                return scheduler['learning_rate']*0.1
            return scheduler['learning_rate'] * np.power(scheduler['decay_rate'], decay)

        def train_step(x_mel, y_sed, y_doa, LR):
            """
            A single training step
            """
            frame_seq_len = np.ones(len(y_sed),dtype=int) * y_sed.shape[1]
            feed_dict = {
              net.input_x_mel: x_mel,
              net.input_y_sed: y_sed,
              net.input_y_doa: y_doa,
              net.dropout_keep_prob_rnn: params['dropout_keep_prob_rnn'],
              net.dropout_keep_prob_cnn: params['dropout_keep_prob_cnn'],
              net.dropout_keep_prob_dnn: params['dropout_keep_prob_dnn'],
              net.frame_seq_len: frame_seq_len,
              net.istraining: 1,
              learning_rate: LR
            }
            _, step, loss_sed, loss_doa, output_loss, total_loss = \
                sess.run([train_op, global_step, net.sed_loss, net.doa_loss, net.output_loss, net.loss], feed_dict)
            return step, loss_sed, loss_doa, output_loss, total_loss

        def dev_step(x_mel, y_sed, y_doa, LR):
            frame_seq_len = np.ones(len(y_sed),dtype=int) * y_sed.shape[1]
            feed_dict = {
                net.input_x_mel: x_mel,
                net.input_y_sed: y_sed,
                net.input_y_doa: y_doa,
                net.dropout_keep_prob_rnn: 1,
                net.dropout_keep_prob_cnn: 1,
                net.dropout_keep_prob_dnn: 1,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0,
                learning_rate: LR
            }
            loss_sed, loss_doa, output_loss, total_loss, \
                score_sed, score_doa = sess.run([net.sed_loss, net.doa_loss, net.output_loss, net.loss,
                                                 net.sed_pred, net.doa_pred], feed_dict)
            ret = dict(loss_sed=loss_sed, loss_doa=loss_doa, output_loss=output_loss,
                       total_loss=total_loss, score_sed=score_sed, score_doa=score_doa)
            return ret

        def evaluate(gen=None, LR=0.0001):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss = 0
            loss_sed = 0
            loss_doa = 0
            total_loss = 0

            N = gen._gen_list[0]._data_size
            score_sed = np.zeros(np.append([len(gen._data_index) * N], gen.get_out_shape_sed()))
            score_doa = np.zeros(np.append([len(gen._data_index) * N], gen.get_out_shape_doa()))

            test_batch_size = params['batch_size'] // 8
            num_batch_per_epoch = np.floor(gen._data_size / test_batch_size).astype(np.uint32)
            test_step = 0
            while test_step < num_batch_per_epoch:
                x_mel, y_sed, y_doa = gen.next_batch_whole(test_batch_size)
                ret = dev_step(x_mel, y_sed, y_doa, LR)
                score_sed[test_step * test_batch_size * N : (test_step + 1) * test_batch_size*N] = ret['score_sed']
                score_doa[test_step * test_batch_size * N: (test_step + 1) * test_batch_size*N] = ret['score_doa']
                loss_sed += ret['loss_sed']
                loss_doa += ret['loss_doa']
                output_loss += ret['output_loss']
                total_loss += ret['total_loss']
                test_step += 1
            if (gen._pointer < len(gen._data_index)):
                _, x_mel, y_sed, y_doa = gen.rest_batch_whole()
                ret = dev_step(x_mel, y_sed, y_doa, LR)
                score_sed[test_step * test_batch_size * N: gen._data_size*N] = ret['score_sed']
                score_doa[test_step * test_batch_size * N: gen._data_size*N] = ret['score_doa']
                loss_sed += ret['loss_sed']
                loss_doa += ret['loss_doa']
                output_loss += ret['output_loss']
                total_loss += ret['total_loss']

            return loss_sed, loss_doa, output_loss, total_loss, score_sed, score_doa

        def metric_dcase2019(gen, sed_pred, doa_pred):
            sed_gt = gen.all_label_sed_2d()
            doa_gt = gen.all_label_doa_2d()
            sed_metric = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, feat_cls.nb_frames_1s())
            doa_metric = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt, sed_pred, sed_gt)
            seld_metric = evaluation_metrics.early_stopping_metric(sed_metric, doa_metric)
            return sed_metric, doa_metric, seld_metric

        def metric_dcase2020(gen, sed_pred, doa_pred):
            sed_gt = gen.all_label_sed_2d()
            doa_gt = gen.all_label_doa_2d()
            cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=gen._Ncat,
                                                                 doa_threshold=params['lad_doa_thresh'])
            pred_dict = feat_cls.regression_label_format_to_output_format(sed_pred, doa_pred)
            gt_dict = feat_cls.regression_label_format_to_output_format(sed_gt, doa_gt)
            pred_blocks_dict = feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
            gt_blocks_dict = feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

            cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
            new_metric = cls_new_metric.compute_seld_scores()
            new_seld_metric = evaluation_metrics.early_stopping_metric(new_metric[:2],new_metric[2:])

            return new_metric, new_seld_metric

        def log_file(filename, loss_sed, loss_doa, output_loss, sed_metric, doa_metric, seld_metric, new_metric, new_seld_metric):
            with open(os.path.join(out_dir, filename), "a") as text_file:
                text_file.write("{:0.5f} {:0.5f} {:0.5f} ".format(loss_sed, loss_doa, output_loss))
                # dcase 2019
                text_file.write("{:0.4f} {:0.2f} ".format(doa_metric[0], doa_metric[1]*100))
                text_file.write("{:0.4f} {:0.2f} ".format(sed_metric[0], sed_metric[1]*100))
                text_file.write("{:0.4f} ".format(seld_metric))
                # dcase 2020
                text_file.write("{:0.4f} {:0.2f} ".format(new_metric[2], new_metric[3] * 100))
                text_file.write("{:0.4f} {:0.2f} ".format(new_metric[0], new_metric[1] * 100))
                text_file.write("{:0.4f}\n".format(new_seld_metric))

        def print_metric(sed_metric, doa_metric, seld_metric, new_metric, new_seld_metric):
            s = '2019 DOA-ER: {:0.4f} '.format(doa_metric[0])
            s += 'Fr-Recall: {:0.2f} '.format(doa_metric[1]*100)
            s += 'SED-ER: {:0.4f} '.format(sed_metric[0])
            s += 'SED-F1: {:0.2f} '.format(sed_metric[1]*100)
            s += 'SELD {:0.4f} '.format(seld_metric)

            s += '2020 DOA-ER: {:0.4f} '.format(new_metric[2])
            s += 'Fr-Recall: {:0.2f} '.format(new_metric[3]*100)
            s += 'SED-ER: {:0.4f} '.format(new_metric[0])
            s += 'SED-F1: {:0.2f} '.format(new_metric[1] * 100)
            s += 'SELD: {:0.4f} '.format(new_seld_metric)
            print(s)

        start_time = time.time()
        # Loop over number of epochs
        train_batches_per_epoch = np.floor(data_gen_train._data_size / params['batch_size']).astype(np.uint32)
        for epoch in range(scheduler['training_epoch']):
            applied_LR = learning_schedule(epoch)
            print("{} Epoch number: {} learning rate {}".format(datetime.now(), epoch + 1, applied_LR))

            step = 0
            while step < train_batches_per_epoch:
                x_mel, y_sed, y_doa = data_gen_train.next_batch(params['batch_size'])
                if FLAGS.augment == 1:
                    x_mel = augment_spec(x_mel)
                train_step_, train_loss_sed_, train_loss_doa_, \
                train_output_loss_, train_total_loss_ = train_step(x_mel, y_sed, y_doa, applied_LR)

                time_str = datetime.now().isoformat()
                print("{}: step {}, loss SED {} DOA {} OUTPUT {} TOTAL {}".
                      format(time_str, train_step_, train_loss_sed_, train_loss_doa_,
                             train_output_loss_, train_total_loss_))
                step += 1
            data_gen_train.reset_pointer()
            data_gen_train.shuffle_data()

        # save the last model
        # save the last model
        current_step = tf.train.global_step(sess, global_step)
        checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("Best model seld metric updated")
        source_file = checkpoint_name
        dest_file = os.path.join(checkpoint_path, 'best_model_seld')
        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
        shutil.copy(source_file + '.index', dest_file + '.index')
        shutil.copy(source_file + '.meta', dest_file + '.meta')

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))