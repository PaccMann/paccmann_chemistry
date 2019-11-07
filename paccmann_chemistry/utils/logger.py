from io import BytesIO

import numpy as np
import scipy.misc
import tensorflow as tf


class Logger:
    """Tensorboard Logger."""

    def __init__(self, logdir):
        """Constructor.

        Args:
            logdir (str): Directory of the event file to be written.
        """
        self.writer = tf.compat.v1.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
        )
        self.writer.add_summary(summary, step)

    def log_scalar(self, scalar_tag, value, global_step):
        """
        Logs scalar to event file.

        Args:
            scalar_tag (str): The name of the scalar to be logged.
            value (torch.Tensor or np.array): The values to be logged.
            global_step (int): The logging step.

        Example:
            `tensorboard = Logger(log_dir)
            x = np.arange(1,101)
            y = 20 + 3 * x + np.random.random(100) * 100
            for i in range(0,100):
                tensorboard.log_scalar('myvalue', y[i], i)`
        """
        summary = tf.Summary()
        summary.value.add(tag=scalar_tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1]
            )
            # Create a Summary value
            img_summaries.append(
                tf.compat.v1.Summary.Value(
                    tag='%s/%d' % (tag, i), image=img_sum
                )
            )

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)]
        )
        self.writer.add_summary(summary, step)
        self.writer.flush()
