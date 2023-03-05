date_name = "Date"
close_name = "Close"
supported_tasks = ('prediction', 'regression')
supported_frameworks = ('pytorch', 'sklearn')

def get_dataloader_from_task(task):
    import pynance
    assert(task in supported_tasks)
    if(task == 'prediction'):
        return pynance.utils.datasets.dataloaders.PredictionDataLoader
    elif(task == 'regression'):
        return pynance.utils.datasets.dataloaders.RegressionDataLoader
    else:
        raise NotImplementedError(f'Task {task} not recognized. Supported tasks: {supported_tasks}')
    
def get_trainer(framework, task):
    import pynance
    assert(task in supported_tasks)
    assert(framework in supported_frameworks)
    if(framework == 'pytorch'):
        return pynance.utils.trainers.TorchTrainer
    elif(framework == 'sklearn'):
        return pynance.utils.trainers.SklearnTrainer
    else:
        raise NotImplementedError(f'Framework {framework} not recognized. Supported framework: {supported_frameworks}')
