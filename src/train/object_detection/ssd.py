
from ..base import Base
import tensorflow as tf
import os
import sys
import time
import yaml
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from .ssd_utils.voc_data import create_batch_generator
from .ssd_utils.anchor import generate_default_boxes
from .ssd_utils.network import create_ssd
from .ssd_utils.losses import create_losses


# 训练模型
class Ssd(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_dir = self.dic_engine['data_dir']
        self.data_year = self.dic_engine['data_year']
        self.data_arch = self.dic_engine['data_arch']
        self.hyperparams = self.dic_engine['hyperparams']
        self.batch_size = self.hyperparams['batch_size']
        self.num_batches = self.hyperparams['num_batches']
        self.neg_ratio = self.hyperparams['neg_ratio']
        self.initial_lr = self.hyperparams['initial_lr']
        self.momentum = self.hyperparams['momentum']
        self.weight_decay = self.hyperparams['weight_decay']
        self.num_classes = self.hyperparams['num_classes']
        self.epochs = self.hyperparams['epochs']
        # self.config_path = self.dic_engine['config_path']
        self.config_path = r'/apps/dev/ai_nlp_testing/src/train/object_detection/ssd_utils/config.yml'

        self.checkpoint_dir = self.dic_engine['_out']
        self.pretrained_type = self.dic_engine['pretrained_type']
        # self.gpu_id = self.dic_engine['gpu_id']

    @tf.function
    def train_step(self, imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
        with tf.GradientTape() as tape:
            confs, locs = ssd(imgs)

            conf_loss, loc_loss = criterion(
                confs, locs, gt_confs, gt_locs)

            loss = conf_loss + loc_loss
            l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
            l2_loss = self.weight_decay * tf.math.reduce_sum(l2_loss)
            loss += l2_loss

        gradients = tape.gradient(loss, ssd.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

        return loss, conf_loss, loc_loss, l2_loss

    def train(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        with open(self.config_path) as f:
            cfg = yaml.load(f)

        try:
            config = cfg[self.data_arch.upper()]
        except AttributeError:
            raise ValueError('Unknown architecture: {}'.format(self.data_arch))

        default_boxes = generate_default_boxes(config)

        batch_generator, val_generator, info = create_batch_generator(
            self.data_dir, self.data_year, default_boxes,
            config['image_size'],
            self.batch_size, self.num_batches,
            mode='train', augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes

        try:
            ssd = create_ssd(self.num_classes, self.data_arch,
                             self.pretrained_type,
                             checkpoint_dir=self.checkpoint_dir)
        except Exception as e:
            print(e)
            print('The program is exiting...')
            sys.exit()

        criterion = create_losses(self.neg_ratio, self.num_classes)

        steps_per_epoch = info['length'] // self.batch_size

        lr_fn = PiecewiseConstantDecay(
            boundaries=[int(steps_per_epoch * self.epochs * 2 / 3),
                        int(steps_per_epoch * self.epochs * 5 / 6)],
            values=[self.initial_lr, self.initial_lr * 0.1, self.initial_lr * 0.01])

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_fn,
            momentum=self.momentum)

        train_log_dir = 'logs/train'
        val_log_dir = 'logs/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(self.epochs):
            avg_loss = 0.0
            avg_conf_loss = 0.0
            avg_loc_loss = 0.0
            start = time.time()
            for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
                loss, conf_loss, loc_loss, l2_loss = self.train_step(
                    imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
                avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
                avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
                if (i + 1) % 50 == 0:
                    print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                        epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

            avg_val_loss = 0.0
            avg_val_conf_loss = 0.0
            avg_val_loc_loss = 0.0
            for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
                val_confs, val_locs = ssd(imgs)
                val_conf_loss, val_loc_loss = criterion(
                    val_confs, val_locs, gt_confs, gt_locs)
                val_loss = val_conf_loss + val_loc_loss
                avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
                avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
                avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)
                tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
                tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

            with val_summary_writer.as_default():
                tf.summary.scalar('loss', avg_val_loss, step=epoch)
                tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
                tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                ssd.save_weights(
                    os.path.join(self.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))

    def dump(self):
        # 保存模型
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.train()
        # self.dump()
