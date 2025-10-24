const sharp = require("sharp");

/**
 * All functions accept Buffer -> return Promise<Buffer> (PNG output)
 */

async function applyBasicTransform(inputBuffer) {
  // auto-levels-ish + sharpen + small resize if huge
  const img = sharp(inputBuffer).rotate();
  const meta = await img.metadata();
  // limit max side to 2000px for safety
  const maxSide = 2000;
  const width = meta.width || null;
  const height = meta.height || null;
  const needResize = Math.max(width || 0, height || 0) > maxSide;

  let pipeline = img;
  if (needResize)
    pipeline = pipeline.resize({
      width: maxSide,
      height: maxSide,
      fit: "inside",
    });

  pipeline = pipeline
    .png()
    .modulate({ saturation: 1.05 }) // color
    .linear(1.05, 0) // slight contrast
    .sharpen();

  return await pipeline.toBuffer();
}

async function applyPencil(inputBuffer) {
  // desaturate + edge detect approximation via convolution
  // create grayscale and overlay edges
  const gray = await sharp(inputBuffer)
    .rotate()
    .resize({ width: 1200, fit: "inside" })
    .grayscale()
    .toBuffer();

  // simple edge detection kernel
  const edgeKernel = {
    width: 3,
    height: 3,
    kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
  };

  const edges = await sharp(gray).convolve(edgeKernel).normalise().toBuffer();

  // blend desaturated base with edges to create pencil-like look
  const base = await sharp(inputBuffer)
    .rotate()
    .resize({ width: 1200, fit: "inside" })
    .modulate({ saturation: 0.15 })
    .toBuffer();

  const composed = await sharp(base)
    .composite([{ input: edges, blend: "multiply" }])
    .png()
    .toBuffer();

  return composed;
}

async function applyOil(inputBuffer) {
  // emulate oil by using blur + increase saturation + posterize-like effect
  const img = sharp(inputBuffer)
    .rotate()
    .resize({ width: 1600, fit: "inside" });

  // posterize: reduce colours -> use quantization via toFormat? sharp doesn't have posterize directly.
  // We'll simulate via median blur (not available) + reduce colours by lowering quality via PNG palette isn't trivial.
  // Simpler: slight blur + increase saturation + enhance
  const out = await img
    .modulate({ saturation: 1.3 })
    .png()
    .blur(1.2)
    .toBuffer();

  return out;
}

module.exports = {
  applyBasicTransform,
  applyPencil,
  applyOil,
};
