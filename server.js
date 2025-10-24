require("dotenv").config();
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { applyBasicTransform, applyPencil, applyOil } = require("./transforms");
const { applyAiTransform } = require("./ai_adapter");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5000;
const MAX_FILE_MB = parseInt(process.env.MAX_FILE_MB || "10", 10);

// multer 설정 (메모리 저장)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

/**
 * POST /transform
 * form-data:
 *  - image: file
 *  - mode: basic | pencil | oil | ai
 *  - prompt: (optional) for ai mode
 */
app.post("/transform", upload.single("image"), async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "image file is required" });

    const mode = (req.body.mode || "basic").toLowerCase();
    const prompt = req.body.prompt || "";

    // buffer -> sharp operations in transforms.js
    let outBuffer;
    if (mode === "basic") {
      outBuffer = await applyBasicTransform(req.file.buffer);
    } else if (mode === "pencil") {
      outBuffer = await applyPencil(req.file.buffer);
    } else if (mode === "oil") {
      outBuffer = await applyOil(req.file.buffer);
    } else if (mode === "ai") {
      // AI 처리 포인트 — ai_adapter.js에서 구현
      // applyAiTransform should return Buffer (image bytes)
      outBuffer = await applyAiTransform(req.file.buffer, { prompt });
      if (!outBuffer)
        return res.status(500).json({ error: "AI transform failed" });
    } else {
      return res.status(400).json({ error: "unsupported mode" });
    }

    // 응답: image/png
    res.set("Content-Type", "image/png");
    res.send(outBuffer);
  } catch (err) {
    console.error("transform err", err);
    res.status(500).json({ error: "internal error", detail: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
