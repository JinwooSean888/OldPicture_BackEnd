require("dotenv").config();
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { applyColorizationTransform } = require("./transforms");

const app = express();
app.use(cors());
const upload = multer({ storage: multer.memoryStorage() });

/**
 * ✅ 이미지 업로드 + AI 컬러 복원
 */
app.post("/api/colorize", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "이미지 파일이 없습니다. (image 필드로 업로드)" });
    }

    const resultBuffer = await applyColorizationTransform(req.file.buffer);

    res.set("Content-Type", "image/jpeg");
    res.send(resultBuffer);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "internal error", detail: error.message });
  }
});

// ✅ 서버 시작
app.listen(process.env.PORT || 5000, () => {
  console.log("✅ Backend server running on port", process.env.PORT || 5000);
});
