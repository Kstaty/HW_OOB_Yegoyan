import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag_index in range(self.num_bags):  # Изменено имя переменной цикла
            # Сохраняем индексы с заменой
            sample_indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(sample_indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set([len(bag) for bag in self.indices_list])) == 1, 'All bags should be of the same length!'
        assert len(self.indices_list[0]) == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag_indices in self.indices_list:  # Изменено имя переменной
            model_instance = model_constructor()  # Изменено имя переменной модели
            # Формируем данные и целевые значения для текущего набора
            train_data, train_target = data[bag_indices], target[bag_indices]
            self.models_list.append(model_instance.fit(train_data, train_target))
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Собираем предсказания от всех обученных моделей
        predictions_matrix = np.array([model.predict(data) for model in self.models_list])
        return predictions_matrix.mean(axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        predictions_storage = [[] for _ in range(len(self.data))]
        # Обрабатываем каждый набор и определяем индексы объектов вне обучения
        for training_indices, model in zip(self.indices_list, self.models_list):
            out_of_bag_indices = set(range(len(self.data))) - set(training_indices)
            for index in out_of_bag_indices:
                # Добавляем предсказания для каждого объекта
                prediction = model.predict(self.data[index].reshape(1, -1))
                predictions_storage[index].append(prediction[0])
        
        self.list_of_predictions_lists = np.array(predictions_storage, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Рассчитываем среднее значение предсказаний, если такие существуют
        self.oob_predictions = np.array([
            None if len(predictions) == 0 else np.mean(predictions)
            for predictions in self.list_of_predictions_lists
        ])
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Вычисляем ошибку только для объектов с предсказаниями
        available_predictions_mask = np.array([p is not None for p in self.oob_predictions])
        mse_targets = self.target[available_predictions_mask]
        mse_predictions = self.oob_predictions[available_predictions_mask]
        return np.mean((mse_targets - mse_predictions) ** 2)
