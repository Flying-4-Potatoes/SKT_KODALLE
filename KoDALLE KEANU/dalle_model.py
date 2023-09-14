DALLE_Klue_Roberta(
  (text_emb): Embedding(32000, 1024, padding_idx=1)
  (image_emb): Embedding(1024, 1024)
  (vae): VQGanVAE(
    (model): VQModel(
      (encoder): Encoder(
        (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (down): ModuleList(
          (0): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (downsample): Downsample(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (1): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (downsample): Downsample(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (2): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (downsample): Downsample(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (3): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (downsample): Downsample(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (4): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList(
              (0): AttnBlock(
                (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
                (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): AttnBlock(
                (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
                (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
        )
        (mid): Module(
          (block_1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (attn_1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (block_2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv_out): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (decoder): Decoder(
        (conv_in): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (mid): Module(
          (block_1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (attn_1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (block_2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (up): ModuleList(
          (0): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (2): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
          )
          (1): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (2): ResnetBlock(
                (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (upsample): Upsample(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (2): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (2): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (upsample): Upsample(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (3): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (2): ResnetBlock(
                (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList()
            (upsample): Upsample(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (4): Module(
            (block): ModuleList(
              (0): ResnetBlock(
                (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (1): ResnetBlock(
                (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (2): ResnetBlock(
                (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (attn): ModuleList(
              (0): AttnBlock(
                (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
                (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): AttnBlock(
                (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
                (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              )
              (2): AttnBlock(
                (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
                (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
                (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (upsample): Upsample(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
        )
        (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (loss): VQLPIPSWithDiscriminator(
        (perceptual_loss): LPIPS(
          (scaling_layer): ScalingLayer()
          (net): vgg16(
            (slice1): Sequential(
              (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
              (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): ReLU(inplace=True)
            )
            (slice2): Sequential(
              (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (6): ReLU(inplace=True)
              (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (8): ReLU(inplace=True)
            )
            (slice3): Sequential(
              (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (11): ReLU(inplace=True)
              (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (13): ReLU(inplace=True)
              (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (15): ReLU(inplace=True)
            )
            (slice4): Sequential(
              (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (18): ReLU(inplace=True)
              (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (20): ReLU(inplace=True)
              (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (22): ReLU(inplace=True)
            )
            (slice5): Sequential(
              (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (25): ReLU(inplace=True)
              (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (27): ReLU(inplace=True)
              (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (29): ReLU(inplace=True)
            )
          )
          (lin0): NetLinLayer(
            (model): Sequential(
              (0): Dropout(p=0.5, inplace=False)
              (1): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
          (lin1): NetLinLayer(
            (model): Sequential(
              (0): Dropout(p=0.5, inplace=False)
              (1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
          (lin2): NetLinLayer(
            (model): Sequential(
              (0): Dropout(p=0.5, inplace=False)
              (1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
          (lin3): NetLinLayer(
            (model): Sequential(
              (0): Dropout(p=0.5, inplace=False)
              (1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
          (lin4): NetLinLayer(
            (model): Sequential(
              (0): Dropout(p=0.5, inplace=False)
              (1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
        )
        (discriminator): NLayerDiscriminator(
          (main): Sequential(
            (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.2, inplace=True)
            (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): LeakyReLU(negative_slope=0.2, inplace=True)
            (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.2, inplace=True)
            (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
            (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (10): LeakyReLU(negative_slope=0.2, inplace=True)
            (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (quantize): VectorQuantizer2(
        (embedding): Embedding(1024, 256)
      )
      (quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (post_quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (transformer): Transformer(
    (layers): ReversibleSequence(
      (blocks): ModuleList(
        (0): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (1): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (2): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (3): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (4): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (5): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (6): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (7): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (8): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (9): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (10): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (11): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (12): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (13): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (14): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (15): ReversibleBlock(
          (f): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): CachedAs(
                      (fn): Attention(
                        (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=512, out_features=1024, bias=True)
                          (1): Dropout(p=0.2, inplace=False)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (g): Deterministic(
            (net): LayerScale(
              (fn): PreNorm(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (norm_out): Identity()
                (fn): CachedAs(
                  (fn): PreShiftToken(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Linear(in_features=1024, out_features=8192, bias=True)
                        (1): GEGLU()
                        (2): Dropout(p=0.2, inplace=False)
                        (3): Linear(in_features=4096, out_features=1024, bias=True)
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
  (norm_by_max): DivideMax()
  (to_logits): Sequential(
    (0): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=1024, out_features=33088, bias=True)
  )
)