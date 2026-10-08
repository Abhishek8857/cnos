import os, sys, glob
import torch
import numpy as np
from src.utils.bbox_utils import CropResizePad
from src.model.utils import Detections, convert_npz_to_json 
from omegaconf import OmegaConf
from PIL import Image
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from src.model.loss import Similarity
from segment_anything.utils.amg import rle_to_mask

import logging
logging.basicConfig(level=logging.INFO)


TEMPLATE_DIR = os.environ.get("CNOS_TEMPLATE_DIR", "/root/cnos-ros/cnos/templates/renders")
NUM_MAX_DETS = int(os.environ.get("CNOS_NUM_MAX_DETS", "1"))
CONF_THRESHOLD = float(os.environ.get("CNOS_CONF_THRESHOLD", "0.5"))



def setup():
    config_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "../../configs"))

    print("Config directory:", config_dir)

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="run_inference")
    
    metric = Similarity()
    logging.info("Initialising Model")
    
    # Initialise model
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")

    logging.info("Initialising templates")
    try:
        template_paths = sorted(glob.glob(f"{TEMPLATE_DIR}/*.png"))
    except Exception as e:
        print(f"An error occured while initialising templates: {e}")
    
    
    boxes, templates = [], []
    for path in template_paths:
        image = Image.open(path)
        boxes.append(image.getbbox())
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)
    
    templates = torch.stack(templates).permute(0, 3, 1, 2)    
    boxes = torch.tensor(np.array(boxes))
    proposal_processor = CropResizePad(OmegaConf.create({"image_size": 224}).image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)

    ref_feats = model.descriptor_model.compute_features(templates, token_name="x_norm_clstoken")
    logging.info(f"Ref feats: {ref_feats.shape}")
    return model, metric, ref_feats

@torch.inference_mode()
def segment (model, metric, ref_feats, rgb_path, out_mask_path):
    try:
        rgb = Image.open(rgb_path).convert("RGB")
        detections = model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        descriptors = model.descriptor_model.forward(np.array(rgb), detections)

        scores = metric(descriptors[:, None, :], ref_feats[None, :, :])
        score_per_detection = torch.mean(torch.topk(scores, k=5, dim=-1)[0], dim=-1)
        scores, index = torch.topk(score_per_detection, k=NUM_MAX_DETS, dim=-1)
        detections.filter(index)
        detections.filter(scores > CONF_THRESHOLD)
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
        detections.to_numpy()

        tmp_save_path = "/tmp/cnos_segmentation"
        detections.save_to_file(0, 0, 0, tmp_save_path, "custom", return_results=False)
        results, _ = convert_npz_to_json(idx=0, list_npz_paths=[tmp_save_path + ".npz"])

        if not results:
            return False

        best = max(results, key=lambda d: d["score"])
        mask = rle_to_mask(best["segmentation"])
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_mask_path)
        return True
    finally:
        torch.cuda.empty_cache()

def main():
    model, metric, ref_feats = setup()
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            rgb_path, out_mask_path = line.split("|")
            ok = segment(model, metric, ref_feats, rgb_path, out_mask_path)
            print(f"OK|{out_mask_path}" if ok else "FAIL|no_detection", flush=True)
        except Exception as e:
            print(f"FAIL|{e}", flush=True)


if __name__ == "__main__":
    main()