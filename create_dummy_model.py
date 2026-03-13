# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Create a YOLOv5n dummy model (random untrained weights), optionally modify the input dimensions,
save as a PyTorch checkpoint, and export to ONNX for edge device deployment evaluation.

Usage:
    $ python create_dummy_model.py                              # defaults: 640x640, 80 classes, ONNX export
    $ python create_dummy_model.py --imgsz 320                 # 320x320 input
    $ python create_dummy_model.py --imgsz 480 640             # 480 (H) x 640 (W) input
    $ python create_dummy_model.py --nc 10 --imgsz 416         # 10 classes, 416x416 input
    $ python create_dummy_model.py --no-export                 # save .pt only, skip ONNX export
    $ python create_dummy_model.py --dynamic --simplify        # dynamic axes + ONNX simplification
    $ python create_dummy_model.py --cfg models/yolov5n.yaml --output yolov5n_dummy.pt

Examples:
    Create a YOLOv5n dummy model with custom input size 320x320 and export to ONNX:
        $ python create_dummy_model.py --imgsz 320 --output yolov5n_dummy_320.pt

    Create a dummy model with non-square input 480x640:
        $ python create_dummy_model.py --imgsz 480 640 --output yolov5n_480x640.pt
"""

import argparse
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import DetectionModel
from utils.general import LOGGER, check_img_size, colorstr, print_args
from utils.torch_utils import select_device


def create_dummy_model(cfg="models/yolov5n.yaml", nc=80, ch=3, device="cpu"):
    """Create a YOLOv5 model with random (untrained) weights from a YAML config file.

    Args:
        cfg (str): Path to the model YAML configuration file. Default is 'models/yolov5n.yaml'.
        nc (int): Number of detection classes. Default is 80 (COCO).
        ch (int): Number of input image channels. Default is 3 (RGB).
        device (str | torch.device): Device to place the model on. Default is 'cpu'.

    Returns:
        models.yolo.DetectionModel: Randomly initialised YOLOv5 detection model in eval mode.

    Examples:
        ```python
        model = create_dummy_model(cfg="models/yolov5n.yaml", nc=80)
        ```
    """
    device = select_device(device)
    LOGGER.info(f"{colorstr('Dummy model:')} creating YOLOv5n from {cfg} with nc={nc}, ch={ch} ...")
    model = DetectionModel(cfg=str(ROOT / cfg) if not Path(cfg).is_absolute() else cfg, ch=ch, nc=nc)
    model = model.to(device).float().eval()
    return model


def save_checkpoint(model, path):
    """Save a YOLOv5 model as a PyTorch checkpoint (.pt) compatible with attempt_load().

    The checkpoint stores the full model object under the key ``'model'`` so that it can be
    loaded directly by :func:`models.experimental.attempt_load` and
    :func:`export.run`.

    Args:
        model (models.yolo.DetectionModel): The model to save.
        path (str | Path): Output file path (should end with ``.pt``).

    Returns:
        Path: Resolved path to the saved checkpoint file.

    Examples:
        ```python
        path = save_checkpoint(model, "yolov5n_dummy.pt")
        ```
    """
    path = Path(path)
    nc = model.yaml["nc"]
    ckpt = {"model": model, "nc": nc, "names": model.names}
    torch.save(ckpt, path)
    from utils.general import file_size

    LOGGER.info(f"{colorstr('Checkpoint:')} saved to {path.resolve()} ({file_size(path):.1f} MB)")
    return path.resolve()


def export_to_onnx(weights, imgsz, dynamic=False, simplify=False, opset=12, device="cpu"):
    """Export a saved YOLOv5 .pt checkpoint to ONNX format.

    Delegates to :func:`export.run` so that the full validation and metadata
    pipeline is reused.

    Args:
        weights (str | Path): Path to the ``.pt`` checkpoint produced by :func:`save_checkpoint`.
        imgsz (list[int]): Image size as ``[height, width]``.
        dynamic (bool): Enable dynamic batch / spatial axes in the ONNX graph. Default is False.
        simplify (bool): Simplify the ONNX graph with onnxslim (``--simplify`` in ``export.py``). Default is False.
        opset (int): ONNX opset version. Default is 12.
        device (str): Device to use during export. Default is ``'cpu'``.

    Returns:
        list[str]: List of exported file paths (as returned by :func:`export.run`).

    Examples:
        ```python
        files = export_to_onnx("yolov5n_dummy.pt", imgsz=[640, 640])
        ```
    """
    import export as exp  # import here to avoid circular dependency at module level

    LOGGER.info(f"{colorstr('ONNX export:')} exporting {weights} with imgsz={imgsz} ...")
    files = exp.run(
        weights=str(weights),
        imgsz=imgsz,
        include=["onnx"],
        device=device,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
    )
    return files


def run(
    cfg="models/yolov5n.yaml",
    nc=80,
    ch=3,
    imgsz=(640, 640),
    output="yolov5n_dummy.pt",
    device="cpu",
    no_export=False,
    dynamic=False,
    simplify=False,
    opset=12,
):
    """Create a YOLOv5n dummy model, save it as a checkpoint, and optionally export to ONNX.

    Args:
        cfg (str): Path to the model YAML config. Default is 'models/yolov5n.yaml'.
        nc (int): Number of detection classes. Default is 80.
        ch (int): Number of input channels. Default is 3.
        imgsz (tuple[int, int]): Input image size as (height, width). Default is (640, 640).
        output (str): Path for the saved ``.pt`` checkpoint. Default is 'yolov5n_dummy.pt'.
        device (str): Device for model creation and export. Default is 'cpu'.
        no_export (bool): Skip ONNX export if True. Default is False.
        dynamic (bool): Use dynamic axes in ONNX export. Default is False.
        simplify (bool): Simplify the ONNX model after export. Default is False.
        opset (int): ONNX opset version. Default is 12.

    Returns:
        tuple[Path, list[str]]: Resolved path to the ``.pt`` checkpoint and list of exported
            ONNX file paths (empty list when ``no_export=True``).

    Examples:
        ```python
        pt_path, onnx_files = run(imgsz=(320, 320), nc=10, output="yolov5n_320.pt")
        ```
    """
    # Normalise imgsz: accept int, tuple, or list and expand to [H, W]
    imgsz = [imgsz] if isinstance(imgsz, int) else list(imgsz)
    imgsz = imgsz * 2 if len(imgsz) == 1 else imgsz

    # Validate that imgsz values are multiples of 32 (model max stride)
    imgsz = [check_img_size(x, s=32) for x in imgsz]

    LOGGER.info(
        f"\n{colorstr('create_dummy_model:')} cfg={cfg}, nc={nc}, ch={ch}, imgsz={imgsz}, "
        f"output={output}, device={device}"
    )

    # 1. Build random-weight model
    model = create_dummy_model(cfg=cfg, nc=nc, ch=ch, device=device)

    # 2. Save checkpoint
    pt_path = save_checkpoint(model, output)

    # 3. Export to ONNX
    onnx_files = []
    if not no_export:
        onnx_files = export_to_onnx(
            weights=pt_path,
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            device=device,
        )

    return pt_path, onnx_files


def parse_opt():
    """Parse command-line arguments for create_dummy_model.py."""
    parser = argparse.ArgumentParser(
        description="Create a YOLOv5n dummy model and export to ONNX for edge device deployment evaluation."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="models/yolov5n.yaml",
        help="model YAML config path (default: models/yolov5n.yaml)",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=80,
        help="number of detection classes (default: 80)",
    )
    parser.add_argument(
        "--ch",
        type=int,
        default=3,
        help="number of input channels (default: 3 for RGB)",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="input image size as H [W] (default: 640 640). Must be multiples of 32.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="yolov5n_dummy.pt",
        help="output .pt checkpoint filename (default: yolov5n_dummy.pt)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device: cpu or cuda device index, e.g. 0 (default: cpu)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="skip ONNX export, save .pt checkpoint only",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="ONNX: enable dynamic batch and spatial axes",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="ONNX: simplify the exported model with onnxslim",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    return parser.parse_args()


def main(opt):
    """Entry point: print args then call run()."""
    print_args(vars(opt))
    run(
        cfg=opt.cfg,
        nc=opt.nc,
        ch=opt.ch,
        imgsz=opt.imgsz,
        output=opt.output,
        device=opt.device,
        no_export=opt.no_export,
        dynamic=opt.dynamic,
        simplify=opt.simplify,
        opset=opt.opset,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
