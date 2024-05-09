num_epochs_seg = 10

# Train the segmentation model
for epoch in range(num_epochs_seg):
    segmentation_model.train()
    running_loss_seg = 0.0

    for batch_idx, batch_seg in enumerate(train_loader_seg):
        images_seg, masks_seg = batch_seg['image'], batch_seg['mask']

        optimizer_seg.zero_grad()

        outputs_seg = segmentation_model(images_seg)

        # Resize masks to match the output size
        masks_seg_resized = nn.functional.interpolate(masks_seg, size=outputs_seg.shape[2:], mode='bilinear', align_corners=True)

        loss_seg = criterion_seg(outputs_seg, masks_seg_resized)
        loss_seg.backward()
        optimizer_seg.step()

        running_loss_seg += loss_seg.item() * images_seg.size(0)

    epoch_loss_seg = running_loss_seg / len(train_dataset_seg)
    print(f'Segmentation Epoch [{epoch + 1}/{num_epochs_seg}], Training Loss: {epoch_loss_seg:.4f}')