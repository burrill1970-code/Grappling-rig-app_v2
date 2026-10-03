"""deep_embedding marker family: pretrained, frozen CNN embeddings of the person crop (CPU, no training).

Weights that could really be downloaded in this sandbox (everything else is blocked by the egress policy: HTTP 403 on
CONNECT for download.pytorch.org, huggingface.co, dl.fbaipublicfiles.com, drive.google.com):

  * ultralytics ImageNet classification weights (GitHub release assets, ultralytics/assets v8.3.0):
        yolo11n-cls.pt / yolo11s-cls.pt / yolo11m-cls.pt ...
    feature = pooled output of the Classify head's 1x1 conv, i.e. the input of the final Linear (penultimate layer).
  * fast-reid person re-identification weights (GitHub release assets, JDAI-CV/fast-reid v0.1.1):
        market_bot_R50.pth  (BoT baseline, ResNet50, last_stride=1, trained on Market-1501 for person ReID)
        market_sbs_R50.pth  (Stronger-Baseline, GeM pooling)
    The checkpoints are loaded with torch.load(weights_only=True) into a torchvision ResNet50 (no fast-reid code needed);
    the feature is the standard fast-reid test feature: pooled layer4 -> BNneck (2048-d).
    The weights come from an external repo's CNN trained on pedestrians (not judo), and nothing is trained here.

Nothing here uses frame number, absolute box position or crop width.  The crop is warped to a FIXED input size (256x128
for ReID, 224x112 for the ImageNet model), so the original crop width is not an input, only appearance is.  The person
mask (when used) only replaces background pixels by a constant grey; it does not use any colour threshold, and nothing
here is darkness-specific.  Instances with an empty region abstain (None).

The weights directory is $DEEP_EMB_WEIGHTS (default below); missing files are fetched from the GitHub URLs above.
"""
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")  # ultralytics must not try to phone home

_WDIR = Path(os.environ.get(
    "DEEP_EMB_WEIGHTS",
    "/tmp/claude-0/-home-user-Grappling-rig-app-v2/9af305d0-b71a-552d-b344-ad580d75284a/scratchpad/w"))
_URLS = {
    "yolo11n-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt",
    "yolo11s-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt",
    "yolo11m-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt",
    "market_bot_R50.pth": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth",
    "market_sbs_R50.pth": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth",
}
FILL_BGR = (114, 114, 114)   # ImageNet-mean-like grey for masked-out background
_MODELS = {}                 # name -> wrapper
_EMB = {}                    # (instance key, model, masked, region) -> dict of raw pooled features
TIMES = []                   # seconds per uncached embedding (for the ms/instance report)


def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


def _weights(name):
    p = _WDIR / name
    if not p.exists():
        _WDIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URLS[name], p)
    return p


def _stripes(fmap, n):
    """Average-pool a [1,C,H,W] map into n horizontal stripes (head / torso / legs for n=3) -> [n,C]."""
    H = fmap.shape[2]
    edges = np.linspace(0, H, n + 1).round().astype(int)
    return np.stack([fmap[0, :, a:max(b, a + 1)].mean((1, 2)).numpy() for a, b in zip(edges[:-1], edges[1:])])


class _Yolo:
    size = (224, 112)  # H, W

    def __init__(self, name):
        from ultralytics import YOLO
        self.net = YOLO(str(_weights(name))).model.float().eval()

    def __call__(self, rgb, wmask=None):  # rgb: float32 HxWx3 in 0..255
        torch = _torch()
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None] / 255.0
        with torch.no_grad():
            for m in self.net.model[:-1]:
                x = m(x)
            head = self.net.model[-1]
            f = head.conv(x)            # [1,1280,h,w] (pre-pool, pre-linear)
        g = f.mean((2, 3))[0].numpy()
        return dict(g=g, mg=g, parts=_stripes(f, 3), mid=x.mean((2, 3))[0].numpy())


class _ReID:
    size = (256, 128)

    def __init__(self, name, gem):
        import torchvision
        torch = _torch()
        ck = torch.load(str(_weights(name)), map_location="cpu", weights_only=True)["model"]
        r = torchvision.models.resnet50()
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)           # last_stride = 1 as in fast-reid
        r.load_state_dict({k[9:]: v for k, v in ck.items() if k.startswith("backbone.")}, strict=False)
        self.r = r.eval()
        self.mean = ck["pixel_mean"].flatten().view(1, 3, 1, 1)
        self.std = ck["pixel_std"].flatten().view(1, 3, 1, 1)
        bn = torch.nn.BatchNorm1d(2048)
        bn.load_state_dict({k[len("heads.bnneck."):]: v for k, v in ck.items() if k.startswith("heads.bnneck.")})
        self.bn = bn.eval()
        self.p = float(ck["heads.pool_layer.p"]) if gem else None

    def _pool(self, f, w=None):
        """GAP (BoT) or GeM (SBS) over the whole map, or weighted by w [1,1,h,w] (person-mask coverage per cell)."""
        if w is None:
            w = f.new_ones((1, 1) + tuple(f.shape[2:]))
        w = w / (w.sum() + 1e-9)
        if self.p is None:
            return (f * w).sum((2, 3))
        return (f.clamp(min=1e-6).pow(self.p) * w).sum((2, 3)).pow(1.0 / self.p)

    def __call__(self, rgb, wmask=None):
        torch = _torch()
        r = self.r
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
        x = (x - self.mean) / self.std
        with torch.no_grad():
            x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
            l3 = r.layer3(r.layer2(r.layer1(x)))
            l4 = r.layer4(l3)
            gb = self.bn(self._pool(l4))
            w = None
            if wmask is not None:  # mask coverage resampled to the layer4 grid; fall back to uniform if almost empty
                w = torch.from_numpy(cv2.resize(wmask.astype(np.float32), (l4.shape[3], l4.shape[2]),
                                                interpolation=cv2.INTER_AREA))[None, None]
                if float(w.sum()) < 1.0:
                    w = None
            mgb = self.bn(self._pool(l4, w))
        return dict(g=gb[0].numpy(), mg=mgb[0].numpy(), parts=_stripes(l4, 3), mid=l3.mean((2, 3))[0].numpy())


def _model(name):
    if name not in _MODELS:
        if name.startswith("yolo"):
            _MODELS[name] = _Yolo(name + "-cls.pt")
        elif name == "reid_bot":
            _MODELS[name] = _ReID("market_bot_R50.pth", gem=False)
        elif name == "reid_sbs":
            _MODELS[name] = _ReID("market_sbs_R50.pth", gem=True)
        else:
            raise ValueError(name)
    return _MODELS[name]


def _key(d):
    return (d["clip"], d["frame"], d["label"], tuple(np.round(d["box"], 1)))


def _region(d, img, region):
    if region == "full":
        return img
    from grappling.color import torso_box  # same torso patch as the baseline (pose keypoints, no colour)
    x1, y1, x2, y2 = d["box"]
    full = np.concatenate([d["box"], [d["conf"]], d["kp"].ravel()])
    tx1, ty1, tx2, ty2, _ = torso_box(full)
    s = img.shape[0] / (y2 - y1)
    a, b, e, f = [int(max(0, v)) for v in ((tx1 - x1) * s, (ty1 - y1) * s, (tx2 - x1) * s, (ty2 - y1) * s)]
    p = img[b:f, a:e]
    return None if p.shape[0] < 8 or p.shape[1] < 8 else p


def embed(d, model, masked=True, region="full"):
    """dict(g=global, parts=[3,C], mid=mid-level GAP) for one instance, cached."""
    k = (_key(d), model, masked, region)
    if k in _EMB:
        return _EMB[k]
    t0 = time.perf_counter()
    net = _model(model)
    c, m = d["crop"]
    img = c.copy()
    if masked:
        img[~m] = FILL_BGR
    img4 = _region(d, np.dstack([img, m.astype(np.uint8) * 255]), region)   # mask rides along as 4th channel
    if img4 is None:
        _EMB[k] = None
        return None
    H, W = net.size if region == "full" else (net.size[0] // 2, net.size[1])
    interp = cv2.INTER_AREA if img4.shape[0] > H else cv2.INTER_LINEAR
    img4 = cv2.resize(img4, (W, H), interpolation=interp)
    img, wmask = img4[..., :3], img4[..., 3].astype(np.float32) / 255.0
    out = net(img[..., ::-1].astype(np.float32), wmask if masked else None)         # BGR -> RGB
    _EMB[k] = out
    TIMES.append(time.perf_counter() - t0)
    return out


def _l2(v):
    v = np.asarray(v, float).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def make(model, masked=True, region="full", parts=(), extra=()):
    """Build a feature_fn.  parts: any of 'g' (global), 'stripes', 'mid'; every block is L2-normalised then concatenated."""
    parts = tuple(parts) or ("g",)

    def fn(d):
        e = embed(d, model, masked, region)
        if e is None:
            return None
        blocks = []
        for p in parts:
            if p in ("g", "mg"):
                blocks.append(_l2(e[p]))
            elif p == "mid":
                blocks.append(_l2(e["mid"]))
            elif p == "stripes":
                blocks.extend(_l2(s) for s in e["parts"])
        for (m2, mk, rg, p2) in extra:  # concatenation with another backbone
            e2 = embed(d, m2, mk, rg)
            if e2 is None:
                return None
            blocks.append(_l2(e2[p2]))
        return np.concatenate(blocks)
    return fn


# --- the variants that were actually evaluated (see the study log); name -> feature_fn --------------------------------
FEATURES = {
    "yolo11n_mask": make("yolo11n", True),
    "yolo11n_raw": make("yolo11n", False),
    "reid_bot_mask": make("reid_bot", True),
    "reid_bot_raw": make("reid_bot", False),
    "reid_sbs_mask": make("reid_sbs", True),
    # stage 2 (refinements of the stage-1 winner, chosen on clip1 block)
    "reid_sbs_stripes": make("reid_sbs", True, parts=("stripes",)),             # head / torso / legs pooled separately
    "reid_sbs_g_mid": make("reid_sbs", True, parts=("g", "mid")),               # + layer3 GAP (more colour / texture)
    "reid_sbs_torso": make("reid_sbs", True, region="torso"),                   # torso patch only (same box as baseline)
    "reid_sbs_mpool": make("reid_sbs", True, parts=("mg",)),                    # layer4 pooled over person-mask cells only
    "fusion_hsv_sbs_stripes": None,                                             # filled in below
    "yolo11m_mask": make("yolo11m", True),                                      # scale check of the ImageNet branch
    "reid_sbs_plus_bot": make("reid_sbs", True, parts=("g",), extra=[("reid_bot", True, "full", "g")]),
}


def _fusion_hsv_sbs_stripes(d):
    """Fixed-weight fusion of the BASELINE HSV torso histogram (sqrt -> L2, uses V = darkness) and the SBS stripes embedding.
    Both blocks have equal squared norm (3), so a cosine metric weights them equally.  No weight was tuned."""
    from tools.marker_baseline import hsv_torso
    h = hsv_torso(d)
    e = FEATURES["reid_sbs_stripes"](d)
    if h is None or e is None:
        return None
    return np.concatenate([np.sqrt(3.0) * _l2(np.sqrt(h)), e])


FEATURES["fusion_hsv_sbs_stripes"] = _fusion_hsv_sbs_stripes
