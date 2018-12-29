"""
This is a support class to parse and initialize pretrained word embeddings
"""

class Embeddings:
    def __init__(self, embeddings_file_path):
        """
        This class can be used to create a unique interface to access
        pretrained word embeddings characteristics after reading them from file
        it also write down metadata file for the embeddings
        :param embeddings_file_path: string
        """
        self.vocabulary = dict()
        self.embeddings = []

        # initializing vocabulary and embeddings
        with open(embeddings_file_path) as embeddings_file:
            for line in embeddings_file.readlines()[1:]:
                line_l = line.split("\t")

                vocab = line_l[0]
                vector = [float(x) for x in line_l[3].split(",")]

                self.vocabulary[vocab] = vector
                self.embeddings.append(vector)

            embeddings_file.close()

        print("[WORD SPACE] embeddings parsed!")


        self.vocab_size = len(self.vocabulary)
        self.embed_dim  = len(self.embeddings[0])

    @staticmethod
    def tensorboard_projector(session, tensor_name, metadata_file_path, model_path, log_dir):
        """
        This static method allow the developer to project embeddings using the
        Tensorboard projector utilities
        :param session: tf.Session (or tf.InteractiveSession)
        :param tensor_name: string
        :param metadata_file_path: string
        :param model_path: string
        :param log_dir: string
        """

        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector

        # initializing tensorboard projector config
        config = projector.ProjectorConfig()
        # setting up config variables
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor_name
        embedding.metadata_path = metadata_file_path
        # saving checkpoint in log
        saver = tf.train.Saver()
        saver.save(session, model_path)
        # writing summary in log
        summary_writer = tf.summary.FileWriter(log_dir)
        projector.visualize_embeddings(summary_writer, config)
