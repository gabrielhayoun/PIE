import logging

class Pipeliner:
    def __init__(self,
                 model_class,
                 dataloader_class,
                 trainer_class,
                 analyser_class=None,
                 ) -> None:
        self.model_class = model_class
        self.dataloader_class = dataloader_class
        self.trainer_class = trainer_class
        self.analyser_class = analyser_class
    
        self.model = None
        self.dataloader = None
        self.trainer = None
        self.analyser = None
    
    def init_model(self, parameters):
        self.model = self.model_class(**parameters)
    
    def init_dataloader(self, parameters):
        self.dataloader = self.dataloader_class(**parameters)

    def init_trainer(self, parameters):
        self.trainer = self.trainer_class(**parameters)

    def init_analyser(self, parameters):
        if(self.analyser_class == None):
            self.analyser = None
            logging.warning('Analyser class is None. Analyser was not created.')
        else:
            self.analyser = self.analyser_class(**parameters)

    def train(self):
        self.trainer.train(self.model, self.dataloader.get_train_data(), self.dataloader.get_valid_data())
    
    def evaluate(self):
        self.trainer.evaluate(self.model, self.dataloader.get_valid_data())

    def analyze(self):
        if(self.analyser is not None):
            self.analyser.make_analysis(self.model, self.dataloader)
        else:
            logging.warning('Analyser is None, analysis not performed.')

    def load_data(self, data, kind):
        self.dataloader.load_data(data, kind)

    # ------------- Inference functions ----------------- #
    def load_model_state(self, path):
        self.model.load(path)

    def load_scaler(self, path):
        self.dataloader.load_scaler(path)

    def predict(self, data, prediction_params):
        self.dataloader.load_data(data, 'infer')
        keys = list(data.keys())
        infer_data = self.dataloader.get_infer_data()
        predictions = self.model.predict(infer_data, **prediction_params)
        return self.dataloader.convert_predictions_to_dict(predictions, keys) # TODO
