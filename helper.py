import torch

def write_weight_grad_stats(values, val_type, step, tb_writer):
    has_nans = torch.isnan(values).any()
    if has_nans:
        linfo(f"************ WARNING: {val_type} contain a nan value at batch # {step}")
    tb_writer.add_histogram(f"{val_type}", values, global_step=step)
    values = values.abs()
    tb_writer.add_scalars(main_tag=f"{val_type} Stats",
                          tag_scalar_dict={'abs_min': torch.min(values),
                                           'abs_max': torch.max(values),
                                           'abs_mean': torch.mean(values),
                                           'abs_std': torch.std(values),
                                           'has_nans': has_nans},
                          global_step=step)


def write_model_params(mod, step, tb_writer):
    weights = []
    grads = []
    for p in mod.parameters():
        weights.append(p.data.flatten())
        if p.grad is not None:
            grads.append(p.grad.flatten())
    weights = torch.cat(weights)
    write_weight_grad_stats(weights, 'Weights', step, tb_writer)
    if len(grads) > 0:
        grads = torch.cat(grads)
        write_weight_grad_stats(grads, 'Gradients', step, tb_writer)
