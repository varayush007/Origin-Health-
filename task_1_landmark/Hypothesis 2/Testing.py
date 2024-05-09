# Testing the model
model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images, landmarks = batch['image'], batch['landmarks']
        outputs = model(images)
        landmarks = landmarks.view(-1, 8)
        loss = criterion(outputs, landmarks.float())
        test_loss += loss.item() * images.size(0)

        print(f'Test Batch [{batch_idx + 1}/{len(test_loader)}], Loss: {loss.item():.4f}')

test_loss /= len(test_dataset)
print(f'Test Loss: {test_loss:.4f}')