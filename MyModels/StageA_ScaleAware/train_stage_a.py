"""
train_stage_a.py
----------------
Stage A: scale-equivariant fine-tuning of the MaRS encoder (LoRA only), before
caption alignment (stage B) and VQA training (stage C).

For each training image we make three views of the same place -- the original
and 2x / 4x coarser copies (approximate multilooking: square, average k x k
blocks, square root, resize back to 512) -- and train MaRS's LoRA so that:

  Term A ("which place?"):  predictor(head(z_s), log2(k_t / k_s)) is closest to
                            head(z_t) of the SAME image among all images of the
                            batch at scale t.
  Term B ("which scale?"):  ... and closer to head(z_t) than to head(z_u), the same
                            image at the remaining scale u (forces MaRS to keep
                            resolution in its features).
  Preservation:             the 1x pooled feature stays close (cosine) to frozen MaRS.

Only MaRS's LoRA is kept afterwards (mars_lora.pth); the head and the predictor
are training-time only. Batches come from one dataset at a time, datasets in
proportion to their number of images.

    python train_stage_a.py --config stage_a_config.yaml [--dry-run]
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

# Reuse MaRS loading and the in-MaRS LoRA from SarEncPlusVicuna, so stages B/C
# install exactly the same LoRA when they load this stage's weights.
SAR_VLM_DIR = Path(__file__).resolve().parent.parent / "SarEncPlusVicuna"
sys.path.insert(0, str(SAR_VLM_DIR))
from model.mars_adapters import install_mars_adapters, mars_adapter_params, save_mars_adapters  # noqa: E402

IMG_SIZE = 512


def log(msg, log_file=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Views: approximate multilooking + augmentation
# ---------------------------------------------------------------------------

def multilook(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Approximate k-look multilooking of an amplitude image x [1, H, W]:
    amplitude -> intensity (x^2), average k x k blocks, -> amplitude (sqrt),
    resized back to H x W so the token grid covers the same ground.
    (Oliver & Quegan 1998: multilooking averages intensity, not amplitude.)
    """
    coarse = F.avg_pool2d((x * x)[None], k).sqrt()
    return F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)[0]


def random_flip_rot(x: torch.Tensor, rng=random) -> torch.Tensor:
    """Same random flip / 90-degree rotation for the whole tensor [..., H, W]."""
    if rng.random() < 0.5:
        x = x.flip(-1)
    if rng.random() < 0.5:
        x = x.flip(-2)
    return torch.rot90(x, rng.randrange(4), dims=(-2, -1))


def make_views(img: torch.Tensor, factors) -> torch.Tensor:
    """img [1, H, W] -> [V, 1, H, W]: the original plus one multilooked view per factor."""
    return torch.stack([img] + [multilook(img, k) for k in factors])


# ---------------------------------------------------------------------------
# Data: unique images per dataset, single-dataset batches
# ---------------------------------------------------------------------------

def unique_images(jsonl_path: str, data_root: str) -> list:
    """Unique image paths referenced by a VQA JSONL (many questions share one image), in file order."""
    with open(jsonl_path) as f:
        rels = dict.fromkeys(json.loads(line)["image"] for line in f if line.strip())
    return [os.path.join(data_root, r) for r in rels]


def load_image(path: str) -> torch.Tensor:
    """Load like the VQA pipeline (dataset.py): first channel, float, resized to 512 x 512."""
    from dataset import load_sar_array  # lazy: needs tifffile / PIL

    x = torch.from_numpy(load_sar_array(path).copy()).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if tuple(x.shape[-2:]) != (IMG_SIZE, IMG_SIZE):
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0]
    return x


class ViewsDataset(torch.utils.data.Dataset):
    """Item (dataset_idx, path) -> [V, 1, 512, 512] views of one image."""

    def __init__(self, items, factors, augment: bool):
        self.items, self.factors, self.augment = items, factors, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = load_image(self.items[i][1])
        if self.augment:
            img = random_flip_rot(img)  # module-level random: PyTorch reseeds it per worker
        return make_views(img, self.factors)


def single_dataset_batches(items, batch_size: int, rng: random.Random) -> list:
    """
    Index batches where every batch comes from one dataset. Each dataset's images
    are shuffled and chopped into full batches; then all batches are shuffled
    together, so datasets appear in proportion to their size.
    """
    by_ds = {}
    for idx, (ds, _) in enumerate(items):
        by_ds.setdefault(ds, []).append(idx)
    batches = []
    for idxs in by_ds.values():
        rng.shuffle(idxs)
        batches += [idxs[i:i + batch_size] for i in range(0, len(idxs) - batch_size + 1, batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# Head, predictor, losses
# ---------------------------------------------------------------------------

def make_head(d_in=1024, d_hidden=1024, d_out=256) -> nn.Module:
    """Projection head h: pooled MaRS feature -> loss space (training only)."""
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class Predictor(nn.Module):
    """Predictor P: (head output of view s, log2 scale ratio t/s) -> guess of view t's head output."""

    def __init__(self, d=256, d_hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, d_hidden), nn.GELU(), nn.Linear(d_hidden, d))

    def forward(self, p, log_ratio: float):
        r = torch.full_like(p[:, :1], log_ratio)
        return self.net(torch.cat([p, r], dim=1))


def scale_losses(p: torch.Tensor, predictor: nn.Module, factors_all, tau: float) -> dict:
    """
    p: [V, N, d] head outputs for V views of N images (view v at scale factors_all[v]).
    Over all ordered view pairs (s, t), s != t:
      Term A: InfoNCE, candidates = all N images at scale t, right = same image.
      Term B: right = same image at t vs wrong = same image at the remaining scale u.
    Returns mean losses and accuracies (Term B accuracy: chance = 0.5).
    """
    V, N, _ = p.shape
    p = F.normalize(p, dim=-1)
    labels = torch.arange(N, device=p.device)
    loss_a = loss_b = acc_a = acc_b = 0.0
    pairs = [(s, t) for s in range(V) for t in range(V) if s != t]
    for s, t in pairs:
        q = F.normalize(predictor(p[s], math.log2(factors_all[t] / factors_all[s])), dim=-1)
        logits_a = q @ p[t].T / tau                                   # [N, N]
        loss_a = loss_a + F.cross_entropy(logits_a, labels)
        # strictly greater than every wrong candidate: a tie must not count as correct
        wrong_a = logits_a.masked_fill(torch.eye(N, dtype=torch.bool, device=p.device), float("-inf"))
        acc_a = acc_a + (logits_a.diagonal() > wrong_a.max(1).values).float().mean()
        u = next(v for v in range(V) if v not in (s, t))
        logits_b = torch.stack([(q * p[t]).sum(-1), (q * p[u]).sum(-1)], dim=1) / tau   # [N, 2]
        loss_b = loss_b + F.cross_entropy(logits_b, torch.zeros_like(labels))
        acc_b = acc_b + (logits_b[:, 0] > logits_b[:, 1]).float().mean()
    n = len(pairs)
    return {"loss_a": loss_a / n, "loss_b": loss_b / n, "acc_a": acc_a / n, "acc_b": acc_b / n}


def train_mode(encoder: nn.Module) -> nn.Module:
    """
    encoder.train(), but with timm's stochastic depth (DropPath) kept off: the
    frozen MaRS weights should give deterministic features, and the preservation
    term compares them with the (eval-mode) frozen copy.
    """
    encoder.train()
    for m in encoder.modules():
        if type(m).__name__ == "DropPath":
            m.eval()
    return encoder


def pooled(encoder, views: torch.Tensor) -> torch.Tensor:
    """views [V, B, 1, H, W] -> pooled MaRS features [V, B, d]."""
    V, B = views.shape[:2]
    z = encoder(views.flatten(0, 1)).mean(dim=1)      # [V*B, d]
    return z.view(V, B, -1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(encoder, frozen, head, predictor, val_sets, factors_all, tau, device, batch_size) -> dict:
    """Per dataset: Term A / Term B accuracy over its whole val subset, and cos(1x, frozen MaRS)."""
    encoder.eval()
    out = {}
    for name, ds in val_sets.items():
        ps, cos = [], []
        for i in range(0, len(ds), batch_size):
            views = torch.stack([ds[j] for j in range(i, min(i + batch_size, len(ds)))], dim=1).to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                z = pooled(encoder, views)
                z0 = frozen(views[0]).mean(dim=1)
            ps.append(head(z.float()))
            cos.append(F.cosine_similarity(z[0].float(), z0.float(), dim=-1))
        m = scale_losses(torch.cat(ps, dim=1), predictor, factors_all, tau)
        out[name] = {"acc_a": float(m["acc_a"]), "acc_b": float(m["acc_b"]), "cos_frozen": float(torch.cat(cos).mean())}
    train_mode(encoder)
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage A: scale-equivariant MaRS LoRA fine-tuning")
    ap.add_argument("--config", "-c", default="stage_a_config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="5 steps, print peak GPU memory, save nothing")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    c_tr, c_loss = cfg["training"], cfg["loss"]

    out_dir = Path(c_tr["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = None if args.dry_run else str(out_dir / "stage_a.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(c_tr.get("seed", 42))
    torch.manual_seed(c_tr.get("seed", 42))

    factors = cfg["views"]["factors"]            # e.g. [2, 4]
    factors_all = [1] + factors

    # --- data ---
    train_items, val_sets = [], {}
    for ds in cfg["datasets"]:
        imgs = unique_images(ds["train_jsonl"], ds["data_root"])
        train_items += [(ds["name"], p) for p in imgs]
        val = unique_images(ds["val_jsonl"], ds["data_root"])
        random.Random(0).shuffle(val)
        val_items = [(ds["name"], p) for p in val[: cfg["validation"]["images_per_dataset"]]]
        val_sets[ds["name"]] = ViewsDataset(val_items, factors, augment=False)
        log(f"{ds['name']}: {len(imgs)} train images, {len(val_items)} val images", log_file)
    train_ds = ViewsDataset(train_items, factors, augment=True)

    # --- model: MaRS + LoRA (trainable), frozen MaRS copy, head, predictor ---
    from model.sar_vlm import build_sar_encoder
    encoder = build_sar_encoder(cfg["model"]["encoder_checkpoint"], freeze=True)
    frozen = build_sar_encoder(cfg["model"]["encoder_checkpoint"], freeze=True).to(device).eval()
    encoder.backbone.requires_grad_(False)
    install_mars_adapters(encoder, lora=True, georope=False, stages=cfg["lora"]["stages"],
                          lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"])
    lora_params = list(mars_adapter_params(encoder).values())
    for p in lora_params:
        p.requires_grad_(True)
    train_mode(encoder.to(device))
    head, predictor = make_head().to(device), Predictor().to(device)

    params = lora_params + list(head.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=float(c_tr["learning_rate"]), weight_decay=0.0)
    steps_per_epoch = len(single_dataset_batches(train_items, c_tr["batch_size"], random.Random(0)))
    total = steps_per_epoch * c_tr["max_epochs"]
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    log(f"{len(train_items)} train images, {steps_per_epoch} steps/epoch, up to {c_tr['max_epochs']} epochs", log_file)

    csv_path = out_dir / "stage_a_losses.csv"
    best_b, bad_epochs, step = -1.0, 0, 0
    for epoch in range(1, c_tr["max_epochs"] + 1):
        batches = single_dataset_batches(train_items, c_tr["batch_size"], rng)
        loader = torch.utils.data.DataLoader(train_ds, batch_sampler=batches,
                                             num_workers=c_tr.get("num_workers", 4),
                                             collate_fn=partial(torch.stack, dim=1))   # -> [V, B, 1, H, W]
        for views in loader:                                       # [V, B, 1, H, W]
            views = views.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                z = pooled(encoder, views)
                with torch.no_grad():
                    z0 = frozen(views[0]).mean(dim=1)
            m = scale_losses(head(z.float()), predictor, factors_all, c_loss["temperature"])
            pres = 1 - F.cosine_similarity(z[0].float(), z0.float(), dim=-1).mean()
            loss = m["loss_a"] + c_loss["lambda_b"] * m["loss_b"] + c_loss["lambda_preserve"] * pres

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if args.dry_run and step >= 5:
                mem = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else "n/a (CPU)"
                log(f"DRY RUN: 5 steps OK, batch {c_tr['batch_size']} x {len(factors_all)} views, "
                    f"peak GPU memory {mem}, loss {loss.item():.4f}")
                return
            if not args.dry_run:
                row = {"epoch": epoch, "step": step, "split": "train", "loss": loss.item(),
                       "loss_a": float(m["loss_a"]), "loss_b": float(m["loss_b"]), "preserve": pres.item(),
                       "acc_a": float(m["acc_a"]), "acc_b": float(m["acc_b"]), "lr": sched.get_last_lr()[0]}
                new = not csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
            if step % 50 == 0:
                log(f"epoch {epoch} step {step}: loss {loss.item():.4f} A {float(m['loss_a']):.3f} "
                    f"B {float(m['loss_b']):.3f} pres {pres.item():.3f} accB {float(m['acc_b']):.3f}", log_file)

        # --- validation, checkpoint, stopping rule ---
        val = evaluate(encoder, frozen, head, predictor, val_sets, factors_all,
                       c_loss["temperature"], device, c_tr["batch_size"])
        acc_b = sum(v["acc_b"] for v in val.values()) / len(val)
        cos0 = sum(v["cos_frozen"] for v in val.values()) / len(val)
        for name, v in val.items():
            log(f"epoch {epoch} VAL {name}: TermA acc {v['acc_a']:.3f}  TermB acc {v['acc_b']:.3f}  "
                f"cos(frozen) {v['cos_frozen']:.3f}", log_file)
        log(f"epoch {epoch} VAL mean: TermB acc {acc_b:.3f}  cos(frozen) {cos0:.3f}", log_file)
        with open(out_dir / "stage_a_val.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "per_dataset": val,
                                "mean_acc_b": acc_b, "mean_cos_frozen": cos0}) + "\n")

        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        save_mars_adapters(encoder, str(ep_dir / "mars_lora.pth"))
        torch.save({"head": head.state_dict(), "predictor": predictor.state_dict()}, ep_dir / "head_predictor.pth")
        if acc_b > best_b:
            best_b, bad_epochs = acc_b, 0
            save_mars_adapters(encoder, str(out_dir / "best_mars_lora.pth"))
            log(f"new best TermB val acc {acc_b:.3f} -> {out_dir / 'best_mars_lora.pth'}", log_file)
        else:
            bad_epochs += 1
            if bad_epochs >= c_tr["patience"]:
                log(f"TermB val acc has not improved for {bad_epochs} epochs; stopping.", log_file)
                break

    verdict = "PROCEED to stage B/C" if best_b >= 0.9 else ("RETHINK" if best_b < 0.7 else "BORDERLINE")
    log(f"Done. Best TermB val acc {best_b:.3f} (target >= 0.90, cos(frozen) >= 0.80): {verdict}", log_file)


if __name__ == "__main__":
    main()
