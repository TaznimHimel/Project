"""Check the remote GPU can actually run this project's training step.

Runs one real forward+backward pass of the exact backbone at the exact batch size
used for training, and reports peak VRAM. Run this before anything else.
"""

import sys


def main() -> int:
    import torch

    print(f"python           : {sys.version.split()[0]}")
    print(f"torch            : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo GPU visible. Training will not be practical. Check the CUDA driver "
              "and that torch was installed from the cu121 index.")
        return 1

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU              : {props.name}")
    print(f"VRAM             : {props.total_memory / 1024**3:.1f} GB")

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(dev)

    # two views per step (original + restyled), as in configurations 2-5
    batch, size = 8, 256
    x = torch.randn(batch * 2, 3, size, size, device=dev)
    y = torch.randint(0, 2, (batch * 2, 1, size, size), device=dev).float()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"peak VRAM (batch {batch}x2 @ {size}px, AMP): {peak:.0f} MB")
    print("\nOK. Environment is ready." if peak < 5200 else
          "\nTight on memory. Lower `train.batch_size` in configs/base.yaml to 6 or 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
