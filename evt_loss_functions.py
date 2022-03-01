def gradient(tensor):
    mid = tensor.shape[1] // 2
    return tensor[:, mid:, :] - tensor[:, :mid, :]

def image_mse_grad(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((gradient(model_output['model_out']) - gradient(gt['img'])) ** 2).mean()}
    else:
        return {'img_loss': (mask * gradient((model_output['model_out']) - gradient(gt['img'])) ** 2).mean()}

