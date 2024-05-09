# Evaluate the segmentation model on the test set
segmentation_model.eval()
test_loss_seg = 0.0

for batch_idx, batch_seg in enumerate(test_loader_seg):
    images_seg, masks_seg = batch_seg['image'], batch_seg['mask']

    outputs_seg = segmentation_model(images_seg)
    outputs_seg_resized  = nn.functional.interpolate(outputs_seg, size=(256, 256), mode='bilinear', align_corners=False)
    loss_seg = criterion_seg(outputs_seg_resized, masks_seg)

    test_loss_seg += loss_seg.item() * images_seg.size(0)

test_loss_seg /= len(test_dataset_seg)
print(f'Segmentation Test Loss: {test_loss_seg:.4f}')