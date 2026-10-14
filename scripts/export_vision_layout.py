"""Export a fixed pretrained classifier under two explicit timing boundaries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from accelerator_bench.runner import save_result
from accelerator_bench.workloads import correctness

IMAGE_URL = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png'
CASES = ('model_nchw', 'model_nhwc', 'image_nchw_first', 'image_nhwc_first')


class VisionLayout(torch.nn.Module):
    def __init__(self, model, case):
        super().__init__()
        if case not in CASES:
            raise ValueError(case)
        self.model, self.case = model, case
        self.register_buffer('mean', torch.tensor([.485, .456, .406]))
        self.register_buffer('std', torch.tensor([.229, .224, .225]))

    def forward(self, x):
        if self.case == 'model_nhwc':
            x = x.permute(0, 3, 1, 2)
        elif self.case == 'image_nchw_first':
            x = x.permute(0, 3, 1, 2)
            x = (x / 255 - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        elif self.case == 'image_nhwc_first':
            x = ((x / 255 - self.mean) / self.std).permute(0, 3, 1, 2)
        return self.model(x)


def state_hash(model):
    """Include parameters as well as buffers; microbenchmark weights use buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export(output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Preserve the existing export; choose a new directory')
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    photo = root / 'astronaut.png'
    if not photo.exists():
        urllib.request.urlretrieve(IMAGE_URL, photo)
    image = Image.open(photo).convert('RGB')
    crop = TF.center_crop(TF.resize(image, 232, InterpolationMode.BILINEAR), [224, 224])
    crop.save(root / 'input-crop.png')
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights).eval()
    normalized = weights.transforms()(image).unsqueeze(0).contiguous()
    pixels = torch.from_numpy(np.array(crop)).unsqueeze(0).float().contiguous()
    normalized_nhwc = normalized.permute(0, 2, 3, 1).contiguous()
    recovered = ((pixels / 255 - torch.tensor([.485, .456, .406])) /
                 torch.tensor([.229, .224, .225])).permute(0, 3, 1, 2)
    torch.testing.assert_close(recovered, normalized, rtol=0, atol=0)
    weights_hash = state_hash(model)
    logical_hash = hashlib.sha256(bytes.fromhex(weights_hash) + normalized.numpy().tobytes()).hexdigest()
    manifest = {'schema_version': 1, 'seed': 1729, 'cases': [],
        'source': 'pretrained_vision_layout', 'model': 'torchvision.mobilenet_v2',
        'weights': weights.name, 'weights_url': weights.url, 'state_sha256': weights_hash,
        'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'image': {'url': IMAGE_URL, 'sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
                  'credit': 'NASA / Eileen Collins; public domain, distributed by scikit-image',
                  'crop_sha256': hashlib.sha256((root/'input-crop.png').read_bytes()).hexdigest()},
        'preprocessing': {'resize': 232, 'crop': [224, 224], 'mean': [.485, .456, .406],
                          'std': [.229, .224, .225], 'interpolation': 'PIL bilinear'},
        'source_precision': 'float32', 'requested_execution_precision': 'float16',
        'excluded': ['decode', 'resize/crop', 'host uint8-to-float32 cast', 'upload', 'camera'],
        'hypothesis_plan': 'research/layout-study/PLAN.md'}
    for case in CASES:
        x = normalized if case == 'model_nchw' else normalized_nhwc if case == 'model_nhwc' else pixels
        module = VisionLayout(copy.deepcopy(model), case).eval()
        with torch.inference_mode():
            reference = copy.deepcopy(module).double()(x.double())
            local_output = module(x)
        model_path, arrays_path = root / f'{case}.onnx', root / f'{case}.npz'
        with torch.inference_mode():
            torch.onnx.export(module, (x,), model_path, input_names=['x'], output_names=['output'],
                              opset_version=17, dynamo=False)
        exported = onnx.load(str(model_path))
        onnx.checker.check_model(exported)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=session_options,
                                       providers=['CPUExecutionProvider'])
        actual = torch.from_numpy(session.run(None, {'x': x.numpy()})[0])
        check = correctness(actual, reference, 'float32')
        if not check['passed']:
            raise RuntimeError(f'ONNX export failed correctness: {case}: {check}')
        top1 = int(reference.argmax(dim=-1).item())
        assert int(local_output.argmax(dim=-1).item()) == top1
        np.savez_compressed(arrays_path, x=x.numpy(), reference=reference.numpy())
        manifest['cases'].append({'case': {'id': case, 'op': 'image_classification',
                                          'shape': list(x.shape)},
            'model': model_path.name, 'arrays': arrays_path.name,
            'input_weight_sha256': logical_hash, 'input_shape': list(x.shape),
            'io_layout': 'nchw' if case == 'model_nchw' else 'nhwc',
            'input_domain': 'normalized' if case.startswith('model_') else 'rgb_0_255',
            'output_kind': 'classification_logits',
            'reference_description': 'CPU FP64 evaluation of this wrapper on original FP32 tensors and pretrained weights',
            'reference_top1': top1, 'reference_label': weights.meta['categories'][top1],
            'onnx_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'arrays_sha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            'onnx_cpu_correctness': check,
            'onnx_op_counts': {op: sum(n.op_type == op for n in exported.graph.node)
                               for op in sorted({n.op_type for n in exported.graph.node})}})
        print(f'checked {case}: top1={top1}, max_error={check["max_abs_error"]:.3g}', flush=True)
    save_result(manifest_path, manifest)
    print(manifest_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    export(**vars(parser.parse_args()))
