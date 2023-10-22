"""
Model training, validation, and testing.
Modifications from starter code:
    1. Save epoch number, training loss, training accuracy, validation loss, & validation accuracy to a csv file.
    2. Compute test loss and accuracy for the model weights with the highest validation accuracy and save to a csv file.
"""

import copy
import numpy as np
import pandas as pd
import torch


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    num_epochs=20,
):
    # Get correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    highest_accuracy = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    def update_model(loader, training):
        current_loss = 0.0
        current_correct = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(training):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                if training:
                    loss.backward()
                    optimizer.step()

            current_loss += loss.item() * inputs.size(0)
            current_correct += torch.sum(preds == labels.data)
        return current_loss, current_correct

    # Progress data
    progress_data = []

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch + 1))

        # train phase
        model.train()
        train_loss, train_correct = update_model(train_loader, True)
        scheduler.step()

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = float(train_correct) / (len(train_loader) * train_loader.batch_size)
        print("Phase: Train  Loss: {} Accuracy: {}".format(epoch_train_loss, epoch_train_accuracy))

        # val phase
        model.eval()
        val_loss, val_correct = update_model(val_loader, False)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = float(val_correct) / (len(val_loader) * val_loader.batch_size)
        print("Phase: Validation  Loss: {} Accuracy: {}".format(epoch_val_loss, epoch_val_accuracy))

        if epoch_val_accuracy > highest_accuracy:
            highest_accuracy = epoch_val_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

        # Progress data
        progress_data.append(
            [
                epoch,
                epoch_train_loss,
                epoch_train_accuracy,
                epoch_val_loss,
                epoch_val_accuracy,
            ]
        )

    # Save progress data to csv
    pd.DataFrame(
        progress_data,
        columns=[
            "epoch",
            "training_loss",
            "training_accuracy",
            "validation_loss",
            "validation_accuracy",
        ],
    ).to_csv(output_dir / "training_progress.csv", index=False)

    # Best model. Compute test loss and accuracy.
    print("Training finished. Highest validation accuracy: {}".format(highest_accuracy))
    model.load_state_dict(best_model_weights)

    model.eval()
    _, test_correct = update_model(test_loader, False)
    test_accuracy = float(test_correct) / (len(test_loader) * test_loader.batch_size)
    with open(output_dir / "test_accuracy.dat", encoding="utf-8", mode="w") as fid:
        fid.write(str(test_accuracy))

    return model
