"""Class to preprocess tabular format data. """
from __future__ import annotations
import datetime
import os
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import model_selection
import os
from preprocessor.tabular import utils
from preprocessor.nnet_survival import nnet_survival
import tensorflow as tf

class TrainerNNetSurvival:
    """NNet Survival Trainer"""

    def __init__(self,
                 input_file_train: Path,
                 input_file_eval: Path,
                 output_dir: Path,
                 unwanted_cols: list[str],
                 numerical_cols: list[str],
                 categorical_cols: list[str],
                 train_batch_size: int = 32,
                 eval_batch_size: int = 100,
                 num_train_examples: int = 50000,
                 num_epochs: int = 10,
                 halflife: float = 1460):

        self.input_file_train = input_file_train
        self.input_file_eval = input_file_eval
        self.output_dir = output_dir
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.unwanted_cols = unwanted_cols
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_examples = num_train_examples
        self.num_epochs = num_epochs
        self.steps_per_epoch = num_train_examples//(train_batch_size*num_epochs)
        self.halflife = halflife
        self.breaks=-np.log(1-np.arange(0.0,0.96,0.05))*self.halflife/np.log(2) 

    def __del__(self):
        print('Deleted')        

    def load_dataset(self, input_file: Path, 
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.EVAL,
                 tab: int = 0):
        """Loads dataset using the tf.data API from CSV files.
        Args:
            pattern: str, file pattern to glob into list of files.
            batch_size: int, the number of examples per batch.
            mode: tf.estimator.ModeKeys to determine if training or evaluating.
        Returns:
            `Dataset` object.
        """        
        if not os.path.exists(input_file):
            raise FileExistsError(f'{input_file} cannot be found.')
        suffix = Path(input_file).suffix
        if suffix not in ['.csv', '.xlsx', '.xls']:
            raise ValueError(f'{suffix} type file not supported yet.')

        raw_data = pd.DataFrame()
        if suffix == '.csv':
            raw_data = pd.read_csv(input_file, header=0)
        elif suffix == '.xlsx':
            raw_data = pd.read_excel(input_file, sheet_name=tab, header=0, index_col=0)

        # clean column names
        raw_data.columns = raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
        # clean string values
        df_obj = raw_data.select_dtypes(['object'])
        raw_data[df_obj.columns] = df_obj.apply(
            lambda x: x.str.strip().str.lower().str.replace(' ', '_'))
        
        # data type
        for c in self.numerical_cols:
            raw_data[c] = raw_data[c].astype('float32')
        for c in self.categorical_cols:
            raw_data[c] = raw_data[c].astype('str')
        
        # calculate the training label for nnet survival
        time = raw_data.pop('rfs')
        event = raw_data.pop('relapse')
        label=nnet_survival.make_surv_array(time,event,self.breaks)

        for c in self.unwanted_cols:
            raw_data.pop(c)
        
        dataset = tf.data.Dataset.from_tensor_slices((dict(raw_data), label))
        
        # Shuffle and repeat for training
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000).repeat().batch(self.train_batch_size)
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.batch(self.eval_batch_size)

        # Take advantage of multi-threading; 1=AUTOTUNE
        dataset = dataset.prefetch(buffer_size=1)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.trainds = dataset
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.evalds = dataset
        
    def create_input_layers(self):
        """Creates dictionary of input layers for each feature.

        Returns:
            Dictionary of `tf.Keras.layers.Input` layers for each feature.
        """
        deep_inputs = {
            colname: tf.keras.layers.Input(
                name=colname, shape=(1,), dtype="float32"
            )
            for colname in self.numerical_cols
        }

        wide_inputs = {
            colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype="string")
            for colname in self.categorical_cols
        }

        inputs = {**wide_inputs, **deep_inputs}
        print(inputs)
        return inputs

    def transform(self, inputs, nembeds):
        """Creates dictionary of transformed inputs.

        Returns:
            Dictionary of transformed Tensors
        """

        deep = {}
        wide = {}
        
        buckets = {
            "age": np.arange(30, 90, 12).tolist(),
            "weight": np.arange(50, 160, 11).tolist(),
        }
        bucketized = {}

        for numerical_column in self.numerical_cols:
            deep[numerical_column] = inputs[numerical_column]
            bucketized[numerical_column] = tf.keras.layers.Discretization(buckets[numerical_column])(inputs[numerical_column])
            wide[f"btk_{numerical_column}"] = tf.keras.layers.CategoryEncoding(
                num_tokens=len(buckets[numerical_column]) + 1, output_mode="one_hot"
            )(bucketized[numerical_column])

        crossed = tf.keras.layers.experimental.preprocessing.HashedCrossing(
            num_bins=len(buckets["age"]) * len(buckets["weight"])
        )((bucketized["age"], bucketized["weight"]))

        deep["age_weight_embeds"] = tf.keras.layers.Flatten()(
            tf.keras.layers.Embedding(
                input_dim=len(buckets["age"])
                * len(buckets["weight"]),
                output_dim=nembeds,
            )(crossed)
        )

        vocab = {
            "centerid": ["True", "False", "Unknown"],
            "gender_m": ["0", "1"],
            "tobacco": ["0", "1", "-1"],
            "alcohol": ["0", "1", "-1"],
            "performance_status": ["0", "1", "2", "3", "4", "-1"],
            "hpv_status": ["0", "1", "-1"],
            "surgery": ["0", "1", "-1"],
            "chemotherapy": ["0", "1"],
        }

        for categorical_column in self.categorical_cols:
            wide[categorical_column] = tf.keras.layers.StringLookup(
                vocabulary=vocab[categorical_column], output_mode="one_hot"
            )(inputs[categorical_column])

        return wide, deep

    def get_model_outputs(self, wide_inputs, deep_inputs, dnn_hidden_units: str = "64 32"):
        """Creates model architecture and returns outputs.

        Args:
            wide_inputs: Dense tensor used as inputs to wide side of model.
            deep_inputs: Dense tensor used as inputs to deep side of model.
            dnn_hidden_units: List of integers where length is number of hidden
                layers and ith element is the number of neurons at ith layer.
        Returns:
            Dense tensor output from the model.
        """
        # Hidden layers for the deep side
        layers = [int(x) for x in dnn_hidden_units.split()]
        deep = deep_inputs
        for layerno, numnodes in enumerate(layers):
            deep = tf.keras.layers.Dense(
                units=numnodes, activation="relu", name=f"dnn_{layerno + 1}"
            )(deep)
        deep_out = deep

        # Linear model for the wide side
        wide_out = tf.keras.layers.Dense(
            units=10, activation="relu", name="linear"
        )(wide_inputs)

        # Concatenate the two sides
        both = tf.keras.layers.Concatenate(name="both")([deep_out, wide_out])

        output1 = tf.keras.layers.Dense(
            units=64, activation="relu", name="dense")(both)
        
        output2 = tf.keras.layers.Dropout(0.2)(output1)
        
        n_intervals = len(self.breaks)-1
        # Final output is a linear activation because this is regression
        output3 = tf.keras.layers.Dense(
            units=n_intervals, activation="sigmoid", name="haz",
            kernel_initializer='zeros', bias_initializer='zeros')(output2)

        return output3

    def build_wide_deep_model(self, dnn_hidden_units="64 32", nembeds=3):
        """Builds wide and deep model using Keras Functional API.

        Returns:
            `tf.keras.models.Model` object.
        """
        # Create input layers
        inputs = self.create_input_layers()

        # transform raw features for both wide and deep
        wide, deep = self.transform(inputs, nembeds)

        # The Functional API in Keras requires: LayerConstructor()(inputs)
        wide_inputs = tf.keras.layers.Concatenate()(wide.values())
        deep_inputs = tf.keras.layers.Concatenate()(deep.values())

        # Get output of model given inputs
        output = self.get_model_outputs(wide_inputs, deep_inputs, dnn_hidden_units)

        # Build model and compile it all together
        model = tf.keras.models.Model(inputs=inputs, outputs=output)

        model.compile(loss=nnet_survival.surv_likelihood(len(self.breaks)-1), 
                      metrics = nnet_survival.surv_likelihood(len(self.breaks)-1),
                      optimizer=tf.keras.optimizers.Adam(),
                      run_eagerly=True)

        self.model = model

    def build_wide_deep_model_test(self):
        """Builds wide and deep model using Keras Functional API.

        Returns:
            `tf.keras.models.Model` object.
        """
        n_intervals = len(self.breaks)-1
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(64, input_shape=(10,),activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.25))

        prop_hazards=0
        if prop_hazards:
            self.model.add(tf.keras.layers.Dense(1, use_bias=0, kernel_initializer='zeros'))
            self.model.add(nnet_survival.PropHazards(n_intervals))
        else:
            self.model.add(tf.keras.layers.Dense(n_intervals, kernel_initializer='zeros', bias_initializer='zeros'))
            self.model.add(tf.keras.layers.Activation('sigmoid'))

        self.model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=tf.keras.optimizers.Adam())

                        
    def train_and_evaluate(self):
        
        self.build_wide_deep_model_test()
        print("Here is our Wide-and-Deep architecture so far:\n")
        print(self.model.summary())
                
        self.load_dataset(
            input_file = self.input_file_train,
            mode = tf.estimator.ModeKeys.TRAIN)

        self.load_dataset(
            input_file = self.input_file_eval, 
            mode = tf.estimator.ModeKeys.EVAL)
        #if args["eval_steps"]:
        #    self.evalds = self.evalds.take(count=args["eval_steps"])

        #checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        #cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #    filepath=checkpoint_path, verbose=1, save_weights_only=True)
        """
        history = self.model.fit(
            self.trainds,
            validation_data=self.evalds,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch
            callbacks=[cp_callback])
        """

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
        history=self.model.fit(self.trainds, 
                               validation_data = self.evalds, 
                               epochs=self.num_epochs, 
                               steps_per_epoch = self.steps_per_epoch,
                               verbose=1, 
                               callbacks=[early_stopping])

""" 
        EXPORT_PATH = os.path.join(
            self.output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        tf.saved_model.save(
            obj=self.model, export_dir=EXPORT_PATH)  # with default serving function
        
        print("Exported trained model to {}".format(EXPORT_PATH)) """