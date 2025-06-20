import os

import torch
from tqdm import tqdm

from src.metrics_impl.BoundingBox import BoundingBox
from src.metrics_impl.BoundingBoxes import BoundingBoxes
from src.metrics_impl.utils import BBType
from hiera_test.train.utils import draw_bboxes


def run_epoch(phase, dataloader,
              num_classes,
              model,
              device,
              optimizer=None,
              scheduler=None,
              bbFormat=None,
              coordType=None, evaluator=None, iouThreshold=None, savePath=None, showPlot=None, bi=None):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_elems_count = 0
    batch_count = len(dataloader)

    all_bounding_boxes = BoundingBoxes()

    cur_tqdm = tqdm(dataloader, total=len(dataloader), leave=False)
    for i, batch in enumerate(cur_tqdm):
        inputs = batch['video'].to(device)  # [B, C, T, H, W]
        boxes = [b.to(device) for b in batch['bbox']]
        labels = [t.to(device) for t in batch['target']]
        detection_results, losses = model(inputs, boxes, labels)

        bz = inputs.shape[0]
        all_elems_count += bz

        show_dict = {}

        if phase == 'train':
            loss = sum(losses.values())

            if isinstance(loss, int):
                print(f"{inputs.shape[0]=}")
                print(f'{loss=} {losses.values()=}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * bz

            show_dict['Loss'] = f'{loss.item():.6f}'
            for k, v in losses.items(): show_dict[k] = v.item()
            cur_tqdm.set_postfix(show_dict)
        elif phase == "test":
            if i == 0:
                bboxes_path = os.path.join(savePath, 'bboxes.jpg')
                draw_bboxes(batch['video'][bi],
                            detection_results[bi]['boxes'].cpu().detach().numpy(),
                            detection_results[bi]['scores'].cpu().detach().numpy(),
                            detection_results[bi]['labels'].cpu().detach().numpy(),
                            bboxes_path)

            for idx in range(len(detection_results)):
                img_id = f"batch{i}_sample{idx}"

                gt_boxes = boxes[idx].bbox.cpu().detach().numpy()
                gt_labels = labels[idx].cpu().detach().argmax(dim=1).numpy()
                for box_idx in range(len(gt_boxes)):
                    x1, y1, x2, y2 = gt_boxes[box_idx]
                    all_bounding_boxes.addBoundingBox(
                        BoundingBox(
                            imageName=img_id,
                            classId=gt_labels[box_idx],
                            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                            bbType=BBType.GroundTruth,
                            format=bbFormat,
                            typeCoordinates=coordType
                        )
                    )

                det_boxes = detection_results[idx]['boxes'].cpu().detach().numpy()
                det_scores = detection_results[idx]['scores'].cpu().detach().numpy()
                det_labels = detection_results[idx]['labels'].cpu().detach().numpy()
                for box_idx in range(len(det_boxes)):
                    x1, y1, x2, y2 = det_boxes[box_idx]
                    all_bounding_boxes.addBoundingBox(
                        BoundingBox(
                            imageName=img_id,
                            classId=det_labels[box_idx],
                            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                            classConfidence=det_scores[box_idx],
                            bbType=BBType.Detected,
                            format=bbFormat,
                            typeCoordinates=coordType
                        )
                    )

            show_dict['bbox_num_per_video'] = sum(r['boxes'].shape[0] for r in detection_results) / bz

            cur_tqdm.set_postfix(show_dict)

    epoch_loss = running_loss / batch_count
    epoch_mAP = 0
    epoch_AP = {f"AP-{i}": 0.0 for i in range(num_classes)}
    prec = {f"precison {i}": 0 for i in range(num_classes)}
    recall = {f"recall {i}": 0 for i in range(num_classes)}
    if phase == 'test' and all_bounding_boxes.getBoundingBoxes():
        metrics = evaluator.GetPascalVOCMetrics(all_bounding_boxes, IOUThreshold=iouThreshold)

        valid_classes = 0
        acc_AP = 0
        for metric_per_class in metrics:
            cl = metric_per_class['class']
            ap = metric_per_class['AP']
            prec[f"precision {cl}"] = metric_per_class["precision"]
            recall[f"recall {cl}"] = metric_per_class["recall"]
            if metric_per_class['total positives'] > 0:
                valid_classes += 1
                acc_AP += ap
                epoch_AP[f'AP-{cl}'] = ap

        epoch_mAP = acc_AP / valid_classes if valid_classes > 0 else 0

    return epoch_loss, epoch_mAP, epoch_AP, prec, recall


def test_epoch(dataloader, *args, **kwargs):
    with torch.inference_mode():
        return run_epoch('test', dataloader, *args, **kwargs)


def train_epoch(dataloader, *args, **kwargs):
    return run_epoch('train', dataloader, *args, **kwargs)


def train_model(dataloaders,
                idx_to_class,
                model=None,
                seed=None,
                writer=None,
                optimizer=None,
                start_epoch=None,
                savePath=None,
                save_frequency=None,
                run_name=None,
                exp_name=None,
                num_epochs=5,
                device=None,
                scheduler=None,
                num_classes=None,
                evaluator=None,
                showPlot=None,
                bi=None,
                bbFormat=None,
                coordType=None,
                iouThreshold=None):
    print(f"Training model with params:")
    print(f"Optim: {optimizer}")
    bboxes_path = os.path.join(savePath, 'bboxes.jpg')
    phases = ['train', 'test']
    for phase_name in dataloaders:
        if phase_name not in phases:
            phases.append(phase_name)

    common_epoch_kwargs = {
        "bbFormat": bbFormat,
        "coordType": coordType,
        "evaluator": evaluator,
        "iouThreshold": iouThreshold,
        "savePath": savePath,
        "showPlot": showPlot,
        "bi": bi
    }

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        for phase in phases:
            current_dataloader = dataloaders[phase]
            if phase == 'train':
                epoch_loss, epoch_mAP, epoch_AP, prec, rec = train_epoch(current_dataloader,
                                                                         num_classes,
                                                                         model,
                                                                         device,
                                                                         optimizer=optimizer,
                                                                         scheduler=scheduler,
                                                                         **common_epoch_kwargs)
            else:
                epoch_loss, epoch_mAP, epoch_AP, prec, rec = test_epoch(current_dataloader,
                                                                        num_classes,
                                                                        model,
                                                                        device,
                                                                        **common_epoch_kwargs)

            if writer:
                if phase == 'train':
                    writer.add_scalar(f'loss/{phase}', epoch_loss, epoch)
                    writer.add_scalar(f"lr/{phase}", optimizer.param_groups[0]['lr'], epoch)
                    writer.add_scalar(f'mAP/{phase}', epoch_mAP, epoch)
                elif phase == 'test':
                    writer.add_scalar(f'mAP/{phase}', epoch_mAP, epoch)
                    writer.add_scalar(f"loss/{phase}", epoch_loss, epoch)
                    # precision_recall_curves = {
                    #     f'images/{i}-curve': os.path.join(savePath, f'{i}.png')
                    #     for i in range(5)
                    # }

                    # image_paths = {
                    #     'images/bboxes': bboxes_path,
                    #     **precision_recall_curves
                    # }

                    # for k, path in image_paths.items():
                    #     if os.path.exists(path):
                    #         img = plt.imread(path)[:, :, :3]
                    #         writer.add_image(k, img, epoch, dataformats='HWC')
                    #         del img
                    for k, v in epoch_AP.items():
                        writer.add_scalar(f'{k}, {idx_to_class[int(k.split("-")[1])]}/{phase}', v, epoch)
                    for k, v in prec.items():
                        writer.add_scalar(f'{k}, {idx_to_class[int(k.split(" ")[1])]}/{phase}', v, epoch)
                    for k, v in rec.items():
                        writer.add_scalar(f'{k}, {idx_to_class[int(k.split(" ")[1])]}/{phase}', v, epoch)

        if epoch % save_frequency == 0:
            model_path = f"runs/{run_name}/{exp_name}_epoch{epoch}.pth"
            print(f"Saving model at {model_path}")
            torch.save({'epoch': epoch,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'model': model.state_dict()}, model_path)
        writer.flush()
