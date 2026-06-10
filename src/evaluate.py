

def evaluate(model,test_ds_prepared, class_names):
    loss, accuracy, top2_accuracy = model.evaluate(test_ds_prepared, verbose=0)