# Validation loop
val_loss = 0.0
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        images, landmarks = batch['image'], batch['landmarks']
        outputs = model(images)
        landmarks = landmarks.view(-1, 8)
        loss = criterion(outputs, landmarks)
        val_loss += loss.item() * images.size(0)
        print(f'Validation Batch [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.4f}')
val_loss /= len(val_dataset)
print(f'Validation Loss: {val_loss:.4f}')