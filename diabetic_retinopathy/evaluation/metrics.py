import tensorflow as tf
import matplotlib.pyplot as plt
import io


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrix,self).__init__(name='confusion_matrix',**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.weight = self.add_weight(name=None, shape=(num_classes,num_classes), initializer="zeros")


    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, labels, predictions, sample_weight=None):

        predictions = tf.argmax(predictions, 1)

        confmat = tf.math.confusion_matrix(labels, predictions, dtype=tf.dtypes.float32, num_classes=self.num_classes)
        self.weight.assign_add(confmat)
        return self.weight

    def result(self):
        confmat = self.weight
        if self.num_classes == 2:

            recall = confmat[1, 1] / ((confmat[1, 1] + confmat[1, 0]) + tf.constant(1e-17))
            precision = confmat[1, 1] / ((confmat[1, 1] + confmat[0, 1]) + tf.constant(1e-17))
            true_negative_rate = confmat[0, 0] / ((confmat[0, 0] + confmat[0, 1]) + tf.constant(1e-17))
            balanced_accuracy = (recall + true_negative_rate) / 2
            f1_score = (2*precision*recall) / (precision+recall)

        else:
            tp = tf.linalg.diag_part(confmat)
            precision = tp / (tf.math.reduce_sum(confmat, 0) + tf.constant(1e-17))
            recall = tp / (tf.math.reduce_sum(confmat, 1) + tf.constant(1e-17))
            balanced_accuracy = tf.reduce_mean(recall)
            f1_score = (2*precision*recall) / (precision+recall + tf.constant(1e-17))

        return recall, precision, f1_score, balanced_accuracy


def plot_ConfusionMatrix(Confmat, num_classes):
    lbl = {0: "No DR",
           1: "DR"}
    lbl = {int(k): v for k, v in lbl.items()}

    # Confmat = np.around(Confmat.astype('float') / (Confmat.sum(axis=1))[:, np.newaxis], decimals=2)
    figure, ax = plt.subplots(figsize=(11, 11))
    plt.imshow(Confmat, cmap=plt.cm.Reds)
    for i in range(num_classes):
        for j in range(num_classes):
            n = Confmat[j, i]
            ax.text(i, j, str(n), va='center', ha='center', size=10, fontdict=None)
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes), ['{} ({})'.format(lbl[i], i) for i in range(num_classes)])
    plt.title("Confusion Matrix")
    plt.colorbar()

    return figure


def plot_to_image(figure):
    memory = io.BytesIO()
    plt.savefig(memory, format='png')
    plt.close(figure)
    memory.seek(0)

    img = tf.image.decode_png(memory.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)

    return img