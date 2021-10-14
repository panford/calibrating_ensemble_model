def test(models, test_loader, loss_fn):
    models.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            losses = torch.empty(len(models), data.shape[0])
            predictions = []
            for i, model in enumerate(models):
                predictions.append(model(data))
                losses[i, :] = loss_fn(predictions[i], target, reduction="sum")

            predictions = torch.stack(predictions)

            loss += torch.mean(losses)
            avg_prediction = predictions.exp().mean(0)

            # get the index of the max log-probability
            class_prediction = avg_prediction.max(1)[1]
            correct += (
                class_prediction.eq(target.view_as(class_prediction)).sum().item()
            )

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct