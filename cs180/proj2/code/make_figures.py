import json
import sys
import time
from pathlib import Path

import numpy as np
import skimage.io as skio
import skimage.transform as sktr
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.signal import convolve2d

import filters as F
import frequencies as Q
from align_image_code import align_images

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MEDIA = ROOT / "media"
PAPER = "#f6f4ee"
FONT_PATH = "/System/Library/Fonts/Supplemental/Georgia.ttf"
PAGE_COLUMN = 758
LABEL_PX = 15
STATS: dict = {}


def read(name: str) -> np.ndarray:
    image = skio.imread(DATA / name)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    return image.astype(np.float64) / 255.0


def resize(image: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(image.shape[:2])
    return sktr.rescale(image, scale, anti_aliasing=True, channel_axis=-1 if image.ndim == 3 else None) if scale < 1 else image


def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8))


def signed(image: np.ndarray, scale: float | None = None) -> np.ndarray:
    scale = scale or np.percentile(np.abs(image), 99.5) + 1e-8
    return np.clip(0.5 + 0.5 * image / scale, 0, 1)


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def colormap(image: np.ndarray, name: str = "magma") -> np.ndarray:
    return colormaps[name](normalize(image))[..., :3]


def wrap(label: str, width: int, font) -> list[str]:
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength
    lines, line = [], ""
    for word in label.split():
        candidate = f"{line} {word}".strip()
        if line and measure(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def grid(items, ncols: int, cell: int = 300, name: str | None = None) -> Image.Image:
    col_w = cell + 20
    width = col_w * ncols + 10
    size = round(LABEL_PX * width / PAGE_COLUMN)
    font, line_h = ImageFont.truetype(FONT_PATH, size), round(size * 1.3)
    tiles = []
    for image, label in items:
        pil = to_pil(image).convert("RGB")
        pil.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        if max(pil.size) < cell * 0.5:
            factor = int(cell * 0.8 // max(pil.size))
            pil = pil.resize((pil.width * factor, pil.height * factor), Image.Resampling.NEAREST)
        tiles.append((pil, wrap(label, cell, font)))
    label_h = max(len(lines) for _, lines in tiles) * line_h + 8
    row_h = max(t.height for t, _ in tiles) + label_h + 16
    nrows = (len(tiles) + ncols - 1) // ncols
    canvas = Image.new("RGB", (width, row_h * nrows + 8), PAPER)
    draw = ImageDraw.Draw(canvas)
    for k, (pil, lines) in enumerate(tiles):
        x0, y0 = 10 + (k % ncols) * col_w, 8 + (k // ncols) * row_h
        for n, line in enumerate(lines):
            draw.text((x0, y0 + 4 + n * line_h), line, fill="#1c1b19", font=font)
        canvas.paste(pil, (x0 + (cell - pil.width) // 2, y0 + label_h))
    if name:
        canvas.save(MEDIA / name, quality=90)
    return canvas


def save(image: np.ndarray, name: str) -> None:
    to_pil(image).save(MEDIA / name, quality=92)


def cameraman() -> np.ndarray:
    return Q.to_gray(read("cameraman.png"))[7:-1, 1:-5]


def part1_1() -> None:
    selfie = resize(read("selfie.jpg")[100:1250], 400)
    me = Q.to_gray(selfie)
    STATS["p11_shape"] = list(me.shape)
    checks = {}
    for label, kernel in [("box9", F.BOX9), ("Dx", F.DX), ("Dy", F.DY)]:
        for mode in ("same", "full"):
            small = me[::4, ::4]
            ref = convolve2d(small, kernel, mode=mode, boundary="fill", fillvalue=0)
            four = F.conv2d_four_loops(small, kernel, mode)
            two = F.conv2d_two_loops(small, kernel, mode)
            checks[f"{label}/{mode}"] = [float(np.abs(four - ref).max()), float(np.abs(two - ref).max())]
    STATS["p11_checks"] = checks

    timings = {}
    for label, fn in [("four", F.conv2d_four_loops), ("two", F.conv2d_two_loops),
                      ("scipy", lambda im, k, m: convolve2d(im, k, mode=m))]:
        start = time.perf_counter()
        out = fn(me, F.BOX9, "same")
        timings[label] = time.perf_counter() - start
    STATS["p11_timings"] = timings

    box = F.conv2d_two_loops(me, F.BOX9, "same")
    dx = F.conv2d_two_loops(me, F.DX, "same")
    dy = F.conv2d_two_loops(me, F.DY, "same")
    grid([(selfie, "Selfie"), (me, "Read as grayscale")], 2, cell=340, name="p11_input.jpg")
    grid([(box, "9x9 box filter"), (signed(dx), "Dx = [1, -1]"), (signed(dy), "Dy = [1, -1]^T")],
         3, cell=320, name="p11_filters.jpg")
    symm = convolve2d(me, F.BOX9, mode="same", boundary="symm")
    crop = np.s_[:90, -90:]
    grid([(box[crop], "zero padding (mine)"), (symm[crop], "scipy boundary='symm'")], 2, cell=260,
         name="p11_boundaries.jpg")


def part1_2() -> None:
    cam = cameraman()
    gx, gy, mag = F.gradients(cam)
    thresholds = [0.12, 0.2, 0.28, 0.36]
    grid([(cam, "Cameraman"), (signed(gx), "dI/dx"), (signed(gy), "dI/dy"), (normalize(mag), "gradient magnitude")],
         2, cell=360, name="p12_gradients.jpg")
    grid([(mag > t, f"threshold {t}") for t in thresholds], 2, cell=360, name="p12_thresholds.jpg")
    STATS["p12_threshold"] = 0.28


def part1_3() -> None:
    cam = cameraman()
    sigma = 2.0
    g = F.gaussian_kernel(sigma)
    smooth = F.conv(cam, g)
    gx, gy, mag = F.gradients(smooth)
    dog_x, dog_y = convolve2d(g, F.DX), convolve2d(g, F.DY)
    gx2, gy2 = F.conv(cam, dog_x), F.conv(cam, dog_y)
    mag2 = np.sqrt(gx2 ** 2 + gy2 ** 2)
    threshold = 0.08
    interior = np.s_[12:-12, 12:-12]
    STATS["p13"] = {"sigma": sigma, "ksize": g.shape[0], "threshold": threshold,
                    "max_diff_interior": float(np.abs(mag - mag2)[interior].max()),
                    "max_diff_all": float(np.abs(mag - mag2).max())}

    _, _, raw_mag = F.gradients(cam)
    grid([(smooth, "Gaussian blurred (sigma 2)"), (signed(gx), "blurred, dI/dx"), (signed(gy), "blurred, dI/dy"),
          (normalize(mag), "gradient magnitude"), (mag > threshold, f"edges, threshold {threshold}"),
          (raw_mag > 0.28, "Part 1.2 edges (no blur)")], 3, cell=280, name="p13_blur_then_diff.jpg")
    grid([(normalize(g), "Gaussian G (13x13)"), (signed(dog_x), "DoG x = G * Dx"), (signed(dog_y), "DoG y = G * Dy")],
         3, cell=220, name="p13_dog_filters.jpg")
    grid([(normalize(mag), "blur, then Dx/Dy"), (normalize(mag2), "single DoG convolution"),
          (mag2 > threshold, "DoG edges"), (np.abs(mag - mag2), f"|difference|, max {np.abs(mag - mag2).max():.0e}")],
         2, cell=360, name="p13_dog_compare.jpg")

    orient = F.orientation_image(gx2, gy2, mag2)
    yy, xx = np.mgrid[-1:1:241j, -1:1:241j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    wheel = F.orientation_image(xx, yy, np.where(r <= 1, r, 0) + 1e-9)
    wheel[r > 1] = 1
    grid([(cam, "Cameraman"), (orient, "orientation (hue) x magnitude (value)"), (wheel, "key: gradient direction")],
         3, cell=300, name="p13_orientation.jpg")


def part2_1() -> None:
    sigma = 2.0
    for key, name, max_side in [("taj", "taj.jpg", 600), ("night", "night.jpg", 700)]:
        im = resize(read(name), max_side)
        low = Q.blur(im, sigma)
        high = im - low
        sharp = Q.sharpen(im, sigma, 1.0)
        grid([(im, "original"), (low, f"blurred (sigma {sigma:g})"), (signed(high, 0.25), "high frequencies (x2, +0.5)"),
              (sharp, "sharpened, alpha = 1")], 2, cell=360 if key == "taj" else 420,
             name=f"p21_{key}.jpg")
        alphas = [0.5, 1, 2, 4]
        grid([(Q.sharpen(im, sigma, a), f"alpha = {a:g}") for a in alphas], 2, cell=360 if key == "taj" else 420,
             name=f"p21_{key}_alphas.jpg")

    building = resize(read("building.jpg"), 900)[180:620, 300:740]
    blurred = Q.blur(building, 2.0)
    resharp = Q.sharpen(blurred, 2.0, 2.0)
    STATS["p21_eval"] = {"mse_blurred": float(np.mean((blurred - building) ** 2)),
                         "mse_resharpened": float(np.mean((resharp - building) ** 2))}
    grid([(building, "original (sharp)"), (blurred, "blurred, sigma 2"), (resharp, "re-sharpened, alpha 2")], 3,
         cell=340, name="p21_eval.jpg")


def aligned_pair(high_name, low_name, high_eyes, low_eyes, width_mult=4.6, up=2.2, down=3.6):
    im_high, im_low = read(high_name), read(low_name)
    pts = (*high_eyes, *low_eyes)
    a, b = align_images(im_high, im_low, pts)
    eye_dist = min(np.hypot(*np.subtract(*low_eyes)), np.hypot(*np.subtract(*high_eyes)))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    half_w = int(width_mult * eye_dist / 2)
    crop = np.s_[max(cy - int(up * eye_dist), 0):cy + int(down * eye_dist), max(cx - half_w, 0):cx + half_w]
    return resize(a[crop], 640), resize(b[crop], 640)


PORTRAIT_EYES = ((185, 555), (360, 545))
BOBBLE_EYES = ((380, 213), (470, 213))
BOBBLE_HI_EYES = ((760, 426), (940, 426))
SELFIE_EYES = ((305, 605), (585, 545))

HYBRIDS = {
    "derek_nutmeg": ("hybrid/hybrid_python/nutmeg.jpg", "hybrid/hybrid_python/DerekPicture.jpg",
                     ((605, 285), (755, 370)), ((295, 345), (440, 330)), 7.0, 9.0, (4.6, 2.2)),
    "bobble_portrait": ("bobble_hi.jpg", "portrait.jpg", BOBBLE_HI_EYES, PORTRAIT_EYES, 4.0, 10.0, (3.0, 2.2)),
    "portrait_selfie": ("portrait.jpg", "selfie.jpg", PORTRAIT_EYES, SELFIE_EYES, 3.0, 8.0, (3.0, 1.9)),
}
FAVORITE = "bobble_portrait"


def part2_2() -> None:
    STATS["p22"] = {}
    for key, (hi, lo, hi_eyes, lo_eyes, s_hi, s_lo, (width, up)) in HYBRIDS.items():
        a, b = aligned_pair(hi, lo, hi_eyes, lo_eyes, width_mult=width, up=up)
        hyb = Q.hybrid_image(a, b, s_hi, s_lo)
        STATS["p22"][key] = {"sigma_high": s_hi, "sigma_low": s_lo,
                             "cut_high": 1 / (2 * np.pi * s_hi), "cut_low": 1 / (2 * np.pi * s_lo),
                             "shape": list(a.shape[:2])}
        save(hyb, f"p22_{key}.jpg")
        if key != FAVORITE:
            grid([(read(hi), "high-frequency source"), (read(lo), "low-frequency source"), (hyb, "hybrid")], 3,
                 cell=300, name=f"p22_{key}_summary.jpg")
        small = [resize(hyb, s) for s in (400, 200, 100, 50)]
        widths = [s.shape[1] for s in small]
        strip = np.ones((small[0].shape[0], sum(widths) + 20 * len(small), 3)) * np.array([246, 244, 238]) / 255
        x = 0
        for s in small:
            strip[strip.shape[0] - s.shape[0]:, x:x + s.shape[1]] = s
            x += s.shape[1] + 20
        save(strip, f"p22_{key}_scales.jpg")

        if key == FAVORITE:
            low, high = Q.low_pass(b, s_lo), Q.high_pass(a, s_hi)
            grid([(read(hi), "original: bobblehead"), (read(lo), "original: dad"), (a, "aligned bobblehead"),
                  (b, "aligned dad")], 4, cell=240, name="p22_fav_inputs.jpg")
            grid([(signed(high, 0.25), f"high pass, sigma {s_hi:g}"), (low, f"low pass, sigma {s_lo:g}"),
                  (hyb, "hybrid")], 3, cell=280, name="p22_fav_process.jpg")
            spectra = [(log_spec(a), "FFT: bobblehead"), (log_spec(b), "FFT: dad"),
                       (log_spec(high), "FFT: high-passed bobble"), (log_spec(low), "FFT: low-passed dad"),
                       (log_spec(hyb), "FFT: hybrid")]
            grid(spectra, 3, cell=300, name="p22_fav_fft.jpg")
            sweep = [(Q.hybrid_image(a, b, s_hi, s), f"sigma_low {s:g}") for s in (4, 7, 10, 16)]
            grid(sweep, 4, cell=240, name="p22_fav_sweep_low.jpg")
            sweep = [(Q.hybrid_image(a, b, s, s_lo), f"sigma_high {s:g}") for s in (1.5, 3, 4, 8)]
            grid(sweep, 4, cell=240, name="p22_fav_sweep_high.jpg")

        if key == "derek_nutmeg":
            ga, gb = Q.to_gray(a), Q.to_gray(b)
            gray3 = lambda im: np.dstack([im] * 3)
            variants = [
                (Q.hybrid_image(gray3(ga), gray3(gb), s_hi, s_lo), "gray high + gray low"),
                (Q.hybrid_image(gray3(ga), b, s_hi, s_lo), "gray high + color low"),
                (Q.hybrid_image(a, gray3(gb), s_hi, s_lo), "color high + gray low"),
                (Q.hybrid_image(a, b, s_hi, s_lo), "color high + color low"),
            ]
            grid(variants, 4, cell=260, name="p22_color.jpg")


def log_spec(image: np.ndarray) -> np.ndarray:
    spec = Q.log_spectrum(image)
    lo, hi = np.percentile(spec, [1, 99.9])
    return colormap(np.clip(spec, lo, hi), "magma")


LEVELS, SIGMA = 6, 2.0


def show_band(band: np.ndarray) -> np.ndarray:
    return normalize(band)


def part2_3_and_2_4() -> None:
    apple, orange = read("spline/apple.jpeg"), read("spline/orange.jpeg")
    ga, la = Q.gaussian_stack(apple, LEVELS, SIGMA), Q.laplacian_stack(apple, LEVELS, SIGMA)
    go, lo = Q.gaussian_stack(orange, LEVELS, SIGMA), Q.laplacian_stack(orange, LEVELS, SIGMA)
    grid([(g, f"G{i}") for i, g in enumerate(ga)] + [(show_band(l) if i < LEVELS - 1 else l, f"L{i}") for i, l in enumerate(la)],
         LEVELS, cell=180, name="p23_apple_stacks.jpg")
    grid([(g, f"G{i}") for i, g in enumerate(go)] + [(show_band(l) if i < LEVELS - 1 else l, f"L{i}") for i, l in enumerate(lo)],
         LEVELS, cell=180, name="p23_orange_stacks.jpg")
    STATS["p23_reconstruction_error"] = float(np.abs(sum(la) - apple).max())

    mask = np.zeros(apple.shape[:2])
    mask[:, : apple.shape[1] // 2] = 1
    result, part_a, part_b, bands, gm = Q.blend(apple, orange, mask, LEVELS, SIGMA)
    rows = []
    for i in (0, 2, 4):
        rows += [(show_band(part_a[i]), f"apple L{i} x mask"), (show_band(part_b[i]), f"orange L{i} x (1-mask)"),
                 (show_band(bands[i]), f"blended L{i}")]
    rows += [(sum(part_a), "masked apple, all levels"), (sum(part_b), "masked orange, all levels"), (result, "oraple")]
    grid(rows, 3, cell=240, name="p23_fig342.jpg")
    grid([(m[..., 0], f"mask G{i}") for i, m in enumerate(gm)], LEVELS, cell=150, name="p24_mask_stack.jpg")
    naive = np.dstack([mask] * 3) * apple + (1 - np.dstack([mask] * 3)) * orange
    grid([(apple, "apple"), (orange, "orange"), (naive, "hard seam"), (result, "multiresolution blend")], 2, cell=340,
         name="p24_oraple_compare.jpg")
    gray_result = Q.blend(Q.to_gray(apple), Q.to_gray(orange), mask, LEVELS, SIGMA)[0]
    grid([(gray_result, "grayscale blend"), (result, "color blend")], 2, cell=300, name="p24_color_compare.jpg")

    custom_blends()


def ellipse_mask(shape, center, axes, feather: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    inside = ((xx - center[0]) / axes[0]) ** 2 + ((yy - center[1]) / axes[1]) ** 2 <= 1
    return inside.astype(np.float64)


def warp_to(image: np.ndarray, src_eyes, dst_eyes, out_shape) -> np.ndarray:
    tform = sktr.SimilarityTransform.from_estimate(np.array(dst_eyes, float), np.array(src_eyes, float))
    return sktr.warp(image, tform, output_shape=out_shape, mode="edge")


def sky_mask(image: np.ndarray) -> np.ndarray:
    bluish = (image[..., 2] - image[..., 0] > 18 / 255) & (image.mean(axis=2) > 140 / 255)
    labels, _ = ndi.label(bluish)
    sky = np.isin(labels, list(set(np.unique(labels[0])) - {0}))
    return ndi.binary_fill_holes(ndi.binary_closing(sky, iterations=3)).astype(np.float64)


def custom_blends() -> None:
    portrait, bobble, bobble_hi = read("portrait.jpg"), read("bobble.jpg"), read("bobble_hi.jpg")

    bobble_w = warp_to(bobble_hi, BOBBLE_HI_EYES, PORTRAIT_EYES, portrait.shape[:2])
    crop = np.s_[300:960, 15:530]
    left, right = portrait[crop], bobble_w[crop]
    mask = np.zeros(left.shape[:2])
    mask[:, : left.shape[1] // 2] = 1
    half, *_ = Q.blend(left, right, mask, 7, 2.0)
    grid([(left, "dad"), (right, "bobblehead (aligned)"), (mask, "mask"), (half, "blend")], 4, cell=260,
         name="p24_half_summary.jpg")

    face_w = warp_to(read("selfie.jpg"), SELFIE_EYES, BOBBLE_EYES, bobble.shape[:2])
    mask = Q.blur(ellipse_mask(bobble.shape[:2], (425, 258), (82, 74)), 3.0)
    bb, part_a, part_b, bands, gm = Q.blend(face_w, bobble, mask, 6, 1.5)
    crop = np.s_[20:620, 150:700]
    grid([(bobble[crop], "bobblehead"), (face_w[crop], "me, warped to its eyes"), (mask[crop], "irregular mask"),
          (bb[crop], "blend")], 2, cell=340, name="p24_bobble_summary.jpg")
    zoom = np.s_[120:420, 270:580]
    rows = []
    for i in (0, 1, 3, 5):
        label = "low-pass" if i == 5 else f"L{i}"
        disp = (lambda x: x) if i == 5 else show_band
        rows += [(disp(part_a[i][zoom]), f"me {label} x mask"), (disp(part_b[i][zoom]), f"bobble {label} x (1-m)"),
                 (disp(bands[i][zoom]), f"blended {label}")]
    rows += [(sum(part_a)[zoom], "masked me"), (sum(part_b)[zoom], "masked bobblehead"), (bb[zoom], "result")]
    grid(rows, 3, cell=240, name="p24_bobble_laplacian.jpg")
    naive = mask[..., None] * face_w + (1 - mask[..., None]) * bobble
    grid([(naive[zoom], "feathered alpha only"), (bb[zoom], "multiresolution blend")], 2, cell=300,
         name="p24_bobble_compare.jpg")

    building, night = read("building.jpg"), read("night.jpg")
    sky_src = sktr.resize(night[0:290, 360:900], (540, building.shape[1]), anti_aliasing=True)
    sky_src = np.pad(sky_src, ((0, building.shape[0] - sky_src.shape[0]), (0, 0), (0, 0)), mode="edge")
    mask = sky_mask(building)
    dusk, *_ = Q.blend(sky_src, building, mask, 4, 1.0)
    dusk_deep, *_ = Q.blend(sky_src, building, mask, 6, 2.0)
    naive = mask[..., None] * sky_src + (1 - mask[..., None]) * building
    grid([(building, "buildings (overcast)"), (sky_src, "dusk sky (night photo)"), (mask, "sky mask"),
          (naive, "hard cut-out"), (dusk, "blend (4 levels)")], 3, cell=300, name="p24_sky_summary.jpg")
    zoom = np.s_[380:620, 180:420]
    grid([(naive[zoom], "hard cut-out"), (dusk_deep[zoom], "6 levels, sigma 2"), (dusk[zoom], "4 levels, sigma 1")], 3,
         cell=260, name="p24_sky_zoom.jpg")

if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    parts = [part1_1, part1_2, part1_3, part2_1, part2_2, part2_3_and_2_4]
    chosen = sys.argv[1:] or [p.__name__ for p in parts]
    stats_path = MEDIA / "stats.json"
    if stats_path.exists():
        STATS.update(json.loads(stats_path.read_text()))
    for part in parts:
        if part.__name__ in chosen:
            part()
    stats_path.write_text(json.dumps(STATS, indent=2))
    print(json.dumps(STATS, indent=2))
