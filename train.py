import os
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from transformers import BertConfig
from tqdm import tqdm
from datetime import datetime
from util import manual_slice, build_dataset, shuffle_and_batch, get_logdir
from collections import defaultdict


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig('bert-base-cased', return_dict=True)
bert = TFBertForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=1
        )


def fit(
        model,
        train_data,
        train_labels,
        n_epochs,
        optimizer,
        loss_fn,
        train_metrics,
        test_metrics,
        val_data,
        val_labels
        ):

    @tf.function
    def update_network(samples, labels):
        with tf.GradientTape() as tape:
            logits = model(samples)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    def setup_loggers():
        logdir = get_logdir('logs/gradient_tape/')
        print('writing all logs to {}'.format(logdir))
        train_log_dir = os.path.join(logdir, 'train')
        test_log_dir = os.path.join(logdir, 'test')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer

    train_logs = defaultdict(list)
    test_logs = defaultdict(list)

    def write_summary(writer, name, metric, step):
        with writer.as_default():
            tf.summary.scalar(name, metric, step=step)
            writer.flush()

    #train_summary_writer, test_summary_writer = setup_loggers()

    @tf.function
    def train():
        metrics = {}
        for data, labels in zip(train_data, train_labels):
            loss, logits = update_network(
                    data,
                    labels)
            #for metric_obj in train_metrics:
            #    metric_obj(labels, logits)
            #    metrics[metric_obj.name] = metric_obj.result()
            #    metric_obj.reset_states()
        # validation summary + logging
        val_logits = model(val_data)
        #test_summary_writer.set_as_default()
        #for metric_obj in test_metrics:
        #    metric_obj(val_labels, val_logits)
        #    metrics[metric_obj.name] = metric_obj.result()
        #    metric_obj.reset_states()

        return metrics

    #train_summary_writer.set_as_default()
    for epoch in tqdm(range(n_epochs)):
        train()
    #train_summary_writer.close()
    return model


if __name__ == '__main__':

    train_ds, val_ds = build_dataset(
            'datasets/titles.csv',
            tokenizer,
            val_size=0.2)
    train_data, train_labels = train_ds['tokens'], train_ds['labels']
    val_data, val_labels = val_ds['tokens'], val_ds['labels']

    batched_data, batched_labels = shuffle_and_batch(
            train_data,
            train_labels,
            16)

    slice_obj = slice(0, 20, 1)
    train_data_short, train_labels_short = manual_slice(
            train_data,
            train_labels,
            slice_obj)
    val_data_short, val_labels_short = manual_slice(
            val_data,
            val_labels,
            slice_obj)

    # try over/under sampling
    bert = fit(
            bert,
            #[train_data_short],
            #[train_labels_short],
            batched_data,
            batched_labels,
            n_epochs=3,
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss_fn=tf.keras.losses.MeanSquaredError(),
            train_metrics=[
                tf.keras.metrics.MeanSquaredError(name='train_mse'),
                ],
            test_metrics=[
                tf.keras.metrics.MeanSquaredError(name='test_mse'),
                ],
            val_data=val_data,
            val_labels=val_labels
            )

