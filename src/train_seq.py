from tqdm import tqdm
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

def run_epoch(model, loader, optimizer, device, epoch, set_type):
    y_true = []
    y_pred = []
    epoch_loss = 0

    if set_type == 'train':
        desc = f'Epoch {epoch:3d} ┬ Train'
    elif set_type == 'valid':
        desc = '          └ Valid'
    elif set_type == 'test':
        desc = '          @ Test'

    for b in tqdm(loader, desc=desc):
        x, y = b
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = corn_loss(out, y, num_classes=3)

        pred = corn_label_from_logits(out)

        y_true += y.tolist()
        y_pred += pred.cpu().tolist()
        epoch_loss += loss.item()

        if set_type == 'train':
            loss.backward()
            optimizer.step()
            
    epoch_loss /= len(loader)

    return y_true, y_pred, epoch_loss