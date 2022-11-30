import tensorflow as tf
import tensorflow_addons as tfa


class Trainer(object):
    def __init__(self, model, training_config):
        self.model = model
        self.training_config = training_config

    def get_loss_fn(self, loss_id="cce"):
        if loss_id == "cce":
            return tf.keras.losses.CategoricalCrossentropy()

    def get_optimizer_fn(self, optimizer_id="adam", learning_rate=1e-4):
        if optimizer_id == "adam":
            return tf.keras.optimizers.Adam(lr=learning_rate)

    def map_metrics(self, metrics_id):
        if metrics_id == "binary-accuracy":
            return tf.keras.metrics.BinaryAccuracy(name="accuracy")
        elif metrics_id == "precision":
            return tf.keras.metrics.Precision(name="precision")
        elif metrics_id == "recall":
            return tf.keras.metrics.Recall(name="recall")
        elif metrics_id == "f1":
            return tfa.metrics.F1Score(num_classes=len(self.training_config["class_names"]), average='micro', name='f1')
        elif metrics_id == "tp":
            return tf.keras.metrics.TruePositives(name='tp')
        elif metrics_id == "fp":
            return tf.keras.metrics.FalsePositives(name='fp')
        elif metrics_id == "tn":
            return tf.keras.metrics.TrueNegatives(name='tn')
        elif metrics_id == "fn":
            return tf.keras.metrics.FalseNegatives(name='fn'),
        elif metrics_id == "auc":
            return tf.keras.metrics.AUC(name='auc')
        elif metrics_id == "prc":
            return tf.keras.metrics.AUC(name='prc', curve='PR')

    def get_metrics_fn(self, metrics_id="precision"):
        metrics = []
        if isinstance(metrics_id, list):
            for metric in metrics_id:
                metrics.append(self.map_metrics(metric))
        else:
            metrics.append(self.map_metrics(metric))
        return metrics

    def get_callbacks(self):
        es = tf.keras.callbacks.EarlyStopping(
            patience=self.training_config["patience"], restore_best_weights=True)
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=self.training_config["tensorboard_dir"])
        chk = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.training_config["checkpoint_dir"], save_best_only=True)
        csv = tf.keras.callbacks.CSVLogger(
            self.training_config["csv_logger"], separator=',', append=False)
        return [es, tb, chk, csv]

    def compile_model(self, model):
        loss = self.get_loss_fn(self.training_config["loss_id"])
        optimizer = self.get_optimizer_fn(
            self.training_config["optimizer_id"], self.training_config["learning_rate"])
        metrics = self.get_metrics_fn(self.training_config["metrics_id"])
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit_model(self, model, train_dg, validation_dg):
        history = model.fit(
            train_dg,
            validation_data=validation_dg,
            epochs=self.training_config["epochs"],
            callbacks=self.get_callbacks(),
        )
        return model, history

    def evaluate_model(self, model, test_dg):
        results = model.evaluate(test_dg)
        return results

    def launch_trainer(self, train_dg, validation_dg=None, test_dg=None):
        model = self.compile_model(self.model)
        model, history = self.fit_model(model, train_dg, validation_dg)

        if test_dg is not None:
            results = model.evaluate(test_dg)

        return model, history
