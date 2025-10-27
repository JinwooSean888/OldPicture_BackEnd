// ✅ 딱 1번만 존재해야 함
const sharp = require("sharp");

// 기본 필터
function applyBasicTransform(buffer) {
  return sharp(buffer).toBuffer();
}

// 연필 스케치 효과
function applyPencilFilter(buffer) {
  return sharp(buffer)
    .greyscale()
    .modulate({
      brightness: 1.1,
      contrast: 1.3,
    })
    .toBuffer();
}

// 유화 효과
function applyOilFilter(buffer) {
  return sharp(buffer)
    .blur(2)
    .modulate({
      saturation: 1.5,
    })
    .toBuffer();
}

module.exports = { applyBasicTransform, applyPencilFilter, applyOilFilter };
