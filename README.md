# Train_Eval_Module
A Train_Eval_Module to operate 10-fold cross-validation on graph datasets.In the test_max mode, the test_dataset equals the val_dataset, the final test_acc is acc on the single epoch that has the best cross-validation accuracy averaged over the 10 folds.In the val_min mode, the val_dataset and test_dataset are different, and the test_acc is the acc on the epoch while the val_loss is the smallest for each folds.The final test_acc is the average of acc over 10 folds.
