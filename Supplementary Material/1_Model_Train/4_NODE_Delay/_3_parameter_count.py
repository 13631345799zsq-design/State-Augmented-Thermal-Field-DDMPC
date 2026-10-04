import torch

from _0_NODE_delay_Model import HISTORY_LENGTH, PhysicsInformedCNN


INPUT_SIZE = (128, 128)


def main():
    model = PhysicsInformedCNN()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print('Parameter summary:')
    print(f'{"Module":<15}{"Parameters":>15}{"Trainable":>15}')
    for module_name, module in model.named_children():
        module_parameters = sum(parameter.numel() for parameter in module.parameters())
        module_trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        print(f'{module_name:<15}{module_parameters:>15,}{module_trainable:>15,}')

    print(f'\nTotal parameters:     {total_parameters:,}')
    print(f'Trainable parameters: {trainable_parameters:,}')
    print(f'Parameter memory:     {total_parameters * 4 / 1024 ** 2:.2f} MiB (float32)')

    time_D_m = torch.ones(1, 1, *INPUT_SIZE)
    temp_m_history = torch.zeros(1, HISTORY_LENGTH, 1, *INPUT_SIZE)
    flux_m = torch.zeros(1, 1, *INPUT_SIZE)
    cooling_m = torch.zeros(1, 1, *INPUT_SIZE)
    envtemp_m_history = torch.zeros(1, HISTORY_LENGTH, 1, *INPUT_SIZE)

    model.eval()
    with torch.no_grad():
        temp_m_next = model(
            time_D_m, temp_m_history, flux_m, cooling_m, envtemp_m_history
        )

    print('\nForward test:')
    print(f'Temperature output shape: {tuple(temp_m_next.shape)}')


if __name__ == '__main__':
    main()
