from PIL import ImageSequence, Image
import numpy as np
from datetime import datetime
import Networks as Nets

__author__ = 'arbellea@post.bgu.ac.il'


def read_multi_tiff(path, start_z=None, stop_z=None):
    """
    path - Path to the multipage-tiff file
    returns images stacked on axis 0
    """
    try:
        img = Image.open(path)
        if start_z is not None:
            img_itr = ImageSequence.Iterator(img)
            img_itr.position = start_z
            images_list = []
            for img_ind, img in enumerate(img_itr):
                images_list.append(img.copy())
                if img_ind + start_z + 1 == stop_z:
                    break
            images = np.stack(images_list, 0).astype(np.float32)
        else:
            images = np.stack(ImageSequence.Iterator(img), 0).astype(np.float32)
        img.close()
        return images
    except Exception:
        return None


def log_print(*args):
    now_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{}:'.format(now_string), *args)


def get_model(model_name: str):
    model = getattr(Nets, model_name)
    return model


def load_model(model_name: str, *args, **kwargs):
    model = get_model(model_name)
    if len(args) > 0 or len(**kwargs) > 0:
        return model(*args, **kwargs)
    else:
        return model


def bbox_crop(img, margin=10):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    cmin = max(0, cmin - margin)
    rmax = min(img.shape[0], rmax + margin)
    cmax = min(img.shape[1], cmax + margin)
    crop = img[rmin:rmax, cmin:cmax]

    return crop, (rmin, rmax, cmin, cmax)


def bbox_fill(img, crop, loc):
    rmin, rmax, cmin, cmax = loc
    img = img.copy()
    img[rmin:rmax, cmin:cmax] = crop
    return img
