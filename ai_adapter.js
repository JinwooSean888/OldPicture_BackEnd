/**
 * This module shows two example approaches:
 * 1) call an external AI HTTP service (e.g., your Python diffusers microservice)
 * 2) call an external API like OpenAI's image edit endpoint (example sketch)
 *
 * IMPORTANT: neither implementation is enabled by default — fill env vars and uncomment code as needed.
 */

const axios = require("axios");
const FormData = require("form-data");

// Option A: Call separate Python AI service (recommended for large models / GPU)
const PY_AI_SERVICE_URL = process.env.PY_AI_SERVICE_URL || ""; // e.g., http://localhost:8000/transform

async function callPythonAiService(imageBuffer, opts = {}) {
  if (!PY_AI_SERVICE_URL) throw new Error("PY_AI_SERVICE_URL not configured");

  const form = new FormData();
  form.append("image", imageBuffer, {
    filename: "upload.png",
    contentType: "image/png",
  });
  if (opts.prompt) form.append("prompt", opts.prompt);

  const headers = form.getHeaders();
  // If your python service requires auth, add here:
  // headers['Authorization'] = `Bearer ${process.env.PY_AI_SERVICE_KEY}`;

  const resp = await axios.post(PY_AI_SERVICE_URL, form, {
    headers,
    responseType: "arraybuffer",
    maxContentLength: 200000000,
  });
  if (resp.status !== 200)
    throw new Error("python ai service error: " + resp.status);
  return Buffer.from(resp.data);
}

// Option B: Example of calling OpenAI Images Edit API (pseudo-code)
// Note: you must include appropriate package / version and follow official docs.
// Here is a minimal fetch-based example (you need to adapt when using an official SDK).
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";

async function callOpenAiImageEdit(imageBuffer, opts = {}) {
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY not configured");

  // This is a placeholder sketch. Use the official OpenAI SDK snippet in production.
  const form = new FormData();
  form.append("image[]", imageBuffer, { filename: "image.png" });
  if (opts.prompt) form.append("prompt", opts.prompt);
  // add other params as required by the API: size, n, etc.

  const resp = await axios.post(
    "https://api.openai.com/v1/images/edits",
    form,
    {
      headers: {
        ...form.getHeaders(),
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      responseType: "json",
    }
  );

  // The real OpenAI Images API returns base64 in JSON or URLs depending on endpoint.
  // Here we assume base64 image in resp.data.data[0].b64_json
  const b64 = resp?.data?.data?.[0]?.b64_json;
  if (!b64) throw new Error("invalid openai response");
  return Buffer.from(b64, "base64");
}

/**
 * Public function used by server.js
 */
async function applyAiTransform(imageBuffer, opts = {}) {
  // Choose preferred strategy:
  // 1) If you have a GPU-hosted python service, prefer calling it:
  if (process.env.PY_AI_SERVICE_URL) {
    return await callPythonAiService(imageBuffer, opts);
  }

  // 2) Otherwise, if you have OpenAI API key, call their image editing API:
  if (process.env.OPENAI_API_KEY) {
    return await callOpenAiImageEdit(imageBuffer, opts);
  }

  throw new Error(
    "No AI backend configured. Set PY_AI_SERVICE_URL or OPENAI_API_KEY"
  );
}

module.exports = { applyAiTransform };
