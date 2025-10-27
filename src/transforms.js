const axios = require("axios");
const FormData = require("form-data");

/**
 * ✅ AI 컬러 복원 (Python Flask API 연동)
 */
async function applyColorizationTransform(imageBuffer) {
  try {
    const formData = new FormData();
    formData.append("image", imageBuffer, { filename: "input.jpg" });

    const response = await axios.post(
      "http://localhost:5001/colorize",
      formData,
      { headers: formData.getHeaders(), responseType: "arraybuffer" }
    );

    return Buffer.from(response.data);
  } catch (error) {
    console.error("❌ Error during Python model processing:", error);
    throw new Error("Colorization AI server error");
  }
}

module.exports = {
  applyColorizationTransform,
};
