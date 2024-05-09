# Training the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images, landmarks = batch['image'], batch['landmarks']

        optimizer.zero_grad()
        outputs = model(images)
        landmarks = landmarks.view(-1, 8)
        loss = criterion(outputs, landmarks.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')