import ipaddress
import pickle
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.utils.url_feature_extraction import URLFeaturesExtraction


MODEL_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "models" / "XGBoostModel.pkl"
FEATURE_COLUMNS = [
    "has_https",
    "url_len",
    "domain_len",
    "path_len",
    "query_len",
    "url_depth",
    "subdomain_count",
    "tld_len",
    "url_has_ipv4",
    "url_has_port",
    "sld_len",
    "sld_has_digit",
    "sld_has_hyphen",
    "sld_token_count",
    "dot_count_domain",
    "hyphen_count_domain_path",
    "underscore_count_path_query",
    "slash_count",
    "digit_count",
    "alphabet_count",
    "spl_char_count",
    "url_entropy",
    "domain_entropy",
    "sld_entropy",
    "path_entropy",
    "domain_token_count",
    "path_token_count",
    "total_tokens",
    "avg_token_length",
]

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

if model.n_features_in_ != len(FEATURE_COLUMNS):
    raise RuntimeError(
        f"The saved model expects {model.n_features_in_} features, "
        f"but the local API is configured for {len(FEATURE_COLUMNS)}."
    )

app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Local URL classification using this project's saved XGBoost model. "
        "The prediction is based on URL structure only; it does not inspect "
        "website content or guarantee that a site is safe.\n\n"
        "[Back to the main checker](/)"
    ),
    version="1.0.0",
    docs_url=None,
    openapi_tags=[
        {
            "name": "Browser pages",
            "description": "Open the checker or review the details of a completed URL assessment.",
        },
        {
            "name": "URL classification",
            "description": "Submit a URL to the local model and inspect its classification result.",
        },
    ],
)

INTERACTIVE_REFERENCE_CSS = """
<style>
  :root { color-scheme: light; font-family: Inter, "Segoe UI", Arial, sans-serif; color: #16243a; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f4f7fb; }
  .secure-header {
    height: 72px; display: flex; align-items: center; justify-content: space-between;
    padding: 0 max(24px, calc((100vw - 1080px) / 2));
    border-bottom: 1px solid #e5ebf3; background: #fff;
  }
  .guide-credit { margin-top: 32px; color: #8793a4; font-size: 12px; text-align: center; }
  .secure-brand { display: flex; align-items: center; gap: 11px; color: #17243a; font-weight: 750; letter-spacing: -.02em; }
  .secure-mark {
    width: 36px; height: 36px; display: grid; place-items: center;
    border-radius: 11px; background: #245bd6; color: white; font-weight: 800;
  }
  .secure-nav { display: flex; gap: 20px; }
  .secure-nav a { color: #245bd6; font-size: 13px; font-weight: 700; text-decoration: none; }
  .secure-nav a:hover { text-decoration: underline; }
  .swagger-ui .topbar { display: none; }
  .reference-hero {
    max-width: 1080px; margin: 0 auto; padding: 38px 20px 5px;
  }
  .reference-kicker {
    color: #245bd6; font-size: 11px; font-weight: 800; letter-spacing: .12em;
  }
  .reference-hero h1 {
    margin: 10px 0 8px; color: #17243a; font-size: clamp(30px, 4vw, 40px);
    letter-spacing: -.045em;
  }
  .reference-hero p {
    max-width: 700px; margin: 0; color: #68778c; font-size: 14px; line-height: 1.7;
  }
  .reference-meta { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 17px; }
  .reference-meta span {
    padding: 6px 10px; border: 1px solid #e0e8f3; border-radius: 99px;
    background: #fff; color: #52647d; font-size: 11px; font-weight: 700;
  }
  .swagger-ui .info { display: none; }
  .swagger-ui { color: #17243a; font-family: Inter, "Segoe UI", Arial, sans-serif; }
  .swagger-ui .wrapper { max-width: 1080px; padding: 0 20px; }
  .swagger-ui .info .title { color: #17243a; font-size: 34px; font-weight: 750; letter-spacing: -.04em; }
  .swagger-ui .info .title small { top: -3px; border: 0; border-radius: 99px; background: #eaf0ff; color: #245bd6; }
  .swagger-ui .info p, .swagger-ui .info li { color: #66758b; font-size: 14px; line-height: 1.7; }
  .swagger-ui .info a { color: #245bd6; }
  .swagger-ui .scheme-container {
    margin: 0 0 20px; padding: 14px 18px; border: 1px solid #e4eaf2;
    border-radius: 13px; background: #fff; box-shadow: 0 8px 24px #152b4808;
  }
  .swagger-ui .opblock-tag-section { margin-bottom: 18px; }
  .swagger-ui .opblock-tag {
    margin: 0; padding: 15px 18px; border-bottom: 1px solid #e7ecf3;
    color: #17243a; font-size: 17px; font-weight: 750;
  }
  .swagger-ui .opblock-tag small { color: #718096; font-size: 12px; font-weight: 400; }
  .swagger-ui .opblock {
    overflow: hidden; margin: 12px 0; border: 1px solid #dce5f0;
    border-radius: 12px; box-shadow: 0 7px 20px #152b4808;
  }
  .swagger-ui .opblock .opblock-summary { padding: 8px 14px; border: 0; }
  .swagger-ui .opblock .opblock-summary-method {
    min-width: 68px; border-radius: 7px; text-shadow: none; font-size: 11px; letter-spacing: .06em;
  }
  .swagger-ui .opblock .opblock-summary-description {
    display: block; color: #53647a; font-size: 13px; line-height: 1.5;
  }
  .swagger-ui .opblock .opblock-summary-path { color: #283951; font-family: Consolas, monospace; }
  .swagger-ui .opblock .opblock-summary-description { color: #53647a; font-size: 13px; }
  .swagger-ui .opblock.opblock-get { border-color: #b8e2c9; background: #f7fcf8; }
  .swagger-ui .opblock.opblock-get .opblock-summary { background: #f0faf4; }
  .swagger-ui .opblock.opblock-get .opblock-summary-path,
  .swagger-ui .opblock.opblock-get .copy-to-clipboard { display: none; }
  .swagger-ui .opblock.opblock-post { border-color: #c5d5fa; background: #f8faff; }
  .swagger-ui .opblock.opblock-post .opblock-summary { background: #eff4ff; }
  .swagger-ui .opblock.opblock-post .opblock-summary-path,
  .swagger-ui .opblock.opblock-post .copy-to-clipboard { display: none; }
  .swagger-ui .opblock-body { background: #fff; }
  .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-external-docs-wrapper,
  .swagger-ui .opblock-title_normal { color: #56667d; font-size: 13px; }
  .swagger-ui .opblock-section-header { background: #f6f8fb; box-shadow: none; }
  .swagger-ui .opblock-section-header h4 { color: #273850; font-size: 13px; }
  .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #53647a; font-size: 12px; }
  .swagger-ui .parameter__name, .swagger-ui .response-col_status { color: #263750; }
  .swagger-ui input[type=text], .swagger-ui textarea, .swagger-ui select {
    border-color: #d6dfeb; border-radius: 7px; box-shadow: none;
  }
  .swagger-ui .btn {
    border-color: #cbd6e5; border-radius: 7px; color: #31445f;
    font-weight: 650; box-shadow: none;
  }
  .swagger-ui .btn.execute {
    border-color: #245bd6; background: #245bd6; color: #fff;
  }
  .swagger-ui .btn.execute:hover { background: #1949b8; }
  .swagger-ui .model-box, .swagger-ui section.models {
    border-color: #e0e7f0; border-radius: 11px; background: #fff;
  }
  .swagger-ui section.models h4 { color: #273850; }
  .swagger-ui .responses-wrapper { margin-top: 18px; }
  @media (max-width: 600px) {
    .secure-header { height: 64px; padding: 0 16px; }
    .secure-nav { gap: 12px; }
    .secure-nav a { font-size: 12px; }
    .reference-hero { padding-top: 28px; }
    .swagger-ui .info .title { font-size: 27px; }
    .swagger-ui .wrapper { padding: 0 12px; }
    .swagger-ui .opblock .opblock-summary { padding: 10px; flex-wrap: wrap; }
    .swagger-ui .opblock .opblock-summary-description {
      display: block; flex: 1 0 calc(100% - 78px); margin: 5px 0 2px 78px;
      white-space: normal;
    }
  }
  .reference-credit {
    max-width: 1080px; margin: 0 auto; padding: 20px;
    color: #8793a4; font-size: 12px; text-align: center;
  }
</style>
"""

API_GUIDE_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>API guide · SecureSight</title>
  <style>
    :root { color-scheme: light; font-family: Inter, "Segoe UI", Arial, sans-serif; color: #16243a; }
    * { box-sizing: border-box; }
    body { min-width: 320px; min-height: 100vh; margin: 0; background: #f4f7fb; }
    header {
      height: 72px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 max(24px, calc((100vw - 1040px) / 2));
      background: #fff; border-bottom: 1px solid #e7ecf3;
    }
    .brand { display: flex; align-items: center; gap: 11px; font-weight: 750; letter-spacing: -.02em; }
    .brand-mark {
      width: 36px; height: 36px; display: grid; place-items: center;
      border-radius: 11px; background: #245bd6; color: white; font-weight: 800;
    }
    a { color: #245bd6; font-weight: 650; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .back-link { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; }
    main { width: min(1040px, calc(100% - 40px)); margin: 52px auto 70px; }
    .eyebrow { color: #245bd6; font-size: 12px; font-weight: 800; letter-spacing: .1em; }
    h1 { margin: 12px 0 10px; font-size: clamp(32px, 5vw, 44px); letter-spacing: -.045em; }
    .intro { max-width: 670px; margin: 0; color: #68778c; font-size: 16px; line-height: 1.7; }
    .section-heading {
      display: flex; align-items: end; justify-content: space-between; gap: 18px;
      margin: 38px 0 15px;
    }
    h2 { margin: 0; font-size: 20px; letter-spacing: -.025em; }
    .section-note { color: #7b8798; font-size: 13px; }
    .guide-credit {
      margin: 30px 0 0; color: #8793a4; font-size: 12px; text-align: center;
    }
    .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    article {
      min-width: 0; padding: 22px; border: 1px solid #e4eaf2; border-radius: 15px;
      background: #fff; box-shadow: 0 10px 28px #1b31500a;
    }
    .card-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .method {
      display: inline-flex; padding: 6px 9px; border-radius: 7px;
      background: #eaf7ef; color: #177245; font-size: 11px;
      font-weight: 850; letter-spacing: .08em;
    }
    .method.post { background: #eaf0ff; color: #245bd6; }
    .route { color: #31425c; font-family: Consolas, "Courier New", monospace; font-size: 13px; }
    article h3 { margin: 17px 0 7px; font-size: 17px; letter-spacing: -.02em; }
    article p { margin: 0; color: #69788e; font-size: 13px; line-height: 1.65; }
    .card-link { display: inline-flex; margin-top: 16px; font-size: 13px; }
    .classify-card { grid-column: 1 / -1; }
    .examples {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px; margin-top: 20px;
    }
    .example { padding: 15px; border: 1px solid #e9edf3; border-radius: 10px; background: #fafbfd; }
    .example h4 { margin: 0 0 9px; font-size: 12px; }
    pre {
      overflow-x: auto; margin: 0; padding: 12px; border-radius: 8px;
      background: #17243a; color: #e5ecf7; font: 12px/1.6 Consolas, "Courier New", monospace;
    }
    .notice {
      margin-top: 22px; padding: 15px 17px; border: 1px solid #f0dfbd;
      border-radius: 11px; background: #fff9ed; color: #79571b;
      font-size: 13px; line-height: 1.6;
    }
    .footer-links {
      display: flex; flex-wrap: wrap; gap: 22px; margin-top: 25px;
      font-size: 13px;
    }
    @media (max-width: 650px) {
      header { height: 64px; padding: 0 18px; }
      main { width: calc(100% - 32px); margin: 36px auto 50px; }
      .cards, .examples { grid-template-columns: 1fr; }
      .classify-card { grid-column: auto; }
      .section-heading { display: block; }
      .section-note { display: block; margin-top: 5px; }
      article { padding: 18px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="brand-mark" aria-hidden="true">S</span><span>SecureSight</span></div>
    <a class="back-link" href="/"><span aria-hidden="true">←</span> Main checker</a>
  </header>
  <main>
    <span class="eyebrow">DEVELOPER GUIDE</span>
    <h1>API overview</h1>
    <p class="intro">A quick guide to the pages and URL-checking endpoint provided by the local SecureSight service. You can use the main checker without using these API routes directly.</p>
    <p class="guide-credit">Local interface customized by Raj Laxmi Kumari</p>

    <div class="section-heading">
      <h2>Browser pages</h2>
      <span class="section-note">Pages you can open in your browser</span>
    </div>
    <section class="cards" aria-label="Browser pages">
      <article>
        <div class="card-top"><span class="method">GET</span><span class="route">/</span></div>
        <h3>Main URL checker</h3>
        <p>Opens the SecureSight page where you enter a website address and request a check.</p>
        <a class="card-link" href="/">Open main checker →</a>
      </article>
      <article>
        <div class="card-top"><span class="method">GET</span><span class="route">/details</span></div>
        <h3>Assessment details</h3>
        <p>Shows the result, model-estimated phishing probability, and explanation after a check.</p>
        <a class="card-link" href="/details">Open details page →</a>
      </article>
    </section>

    <div class="section-heading">
      <h2>URL classification</h2>
      <span class="section-note">The checker uses this endpoint to get a prediction</span>
    </div>
    <section class="cards" aria-label="URL classification endpoint">
      <article class="classify-card">
        <div class="card-top"><span class="method post">POST</span><span class="route">/predict</span></div>
        <h3>Classify a URL</h3>
        <p>Send a URL to the local model. It analyzes URL structure only; it does not open the site or inspect its contents.</p>
        <div class="examples">
          <div class="example">
            <h4>Request example</h4>
            <pre>{"url": "https://example.com/login"}</pre>
          </div>
          <div class="example">
            <h4>Response example</h4>
            <pre>{"probability": 0.08, "prediction": false, "result": "Legitimate"}</pre>
          </div>
        </div>
      </article>
    </section>
    <p class="notice">A “Legitimate” result is not a guarantee that a site is safe. The model can make mistakes and should be treated as an estimate.</p>
    <nav class="footer-links" aria-label="Additional links">
      <a href="/">← Back to the main checker</a>
      <a href="/api-reference">Open interactive API reference</a>
      <a href="/openapi.json">View OpenAPI schema</a>
    </nav>
    <footer class="guide-credit">SecureSight · Local interface customized by Raj Laxmi Kumari</footer>
  </main>
</body>
</html>"""

HOME_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phishing URL Checker</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: #13233b;
      background: #f4f7fb;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
    }
    * { box-sizing: border-box; }
    body { min-width: 320px; min-height: 100vh; margin: 0; }
    .topbar {
      height: 76px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 max(28px, calc((100vw - 1160px) / 2));
      background: #fff; border-bottom: 1px solid #e8edf4;
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 750; letter-spacing: -.02em; }
    .brand-mark {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 12px; color: white; background: #245bd6; font-size: 19px;
      box-shadow: 0 5px 12px #245bd633;
    }
    .top-link { color: #53647c; text-decoration: none; font-size: 14px; font-weight: 600; }
    .top-link:hover { color: #245bd6; }
    body {
      background:
        radial-gradient(ellipse at 16% 12%, #dfeaff 0, transparent 36%),
        radial-gradient(ellipse at 92% 42%, #e1f5ef 0, transparent 30%),
        #f5f7fb;
    }
    .layout { width: min(1160px, calc(100% - 48px)); margin: 0 auto; }
    .hero { padding: 74px 0 34px; text-align: center; }
    .eyebrow {
      display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px;
      border: 1px solid #dce7fb; border-radius: 999px; background: #ffffffb8;
      color: #245bd6; font-size: 12px; font-weight: 750; letter-spacing: .08em;
      text-transform: uppercase;
    }
    .eyebrow-dot { width: 7px; height: 7px; border-radius: 50%; background: #30a879; }
    h1 { margin: 21px 0 12px; font-size: clamp(34px, 5vw, 53px); line-height: 1.08; letter-spacing: -.045em; }
    .hero-copy { max-width: 570px; margin: 0 auto; color: #66758b; font-size: 17px; line-height: 1.65; }
    .checker-card {
      width: min(100%, 790px); margin: 30px auto 0; padding: 30px;
      background: #fff; border: 1px solid #e6ebf2; border-radius: 18px;
      box-shadow: 0 22px 65px #14294d12;
    }
    label { display: block; margin-bottom: 10px; color: #263750; font-size: 14px; font-weight: 700; text-align: left; }
    .input-row {
      min-height: 60px; display: flex; gap: 12px; padding: 6px;
      border: 1px solid #d9e1ec; border-radius: 12px; background: #fbfcfe;
      transition: border-color .18s, box-shadow .18s;
    }
    .input-row:focus-within { border-color: #5680e2; box-shadow: 0 0 0 4px #245bd615; }
    input {
      min-width: 0; flex: 1; padding: 0 14px; border: 0; background: transparent;
      color: #17243a; font: inherit; outline: none;
    }
    input::placeholder { color: #98a5b7; }
    button {
      padding: 0 22px; border: 0; border-radius: 9px; background: #245bd6;
      color: white; font: inherit; font-size: 14px; font-weight: 700;
      cursor: pointer; box-shadow: 0 4px 10px #245bd62a;
      transition: background .18s, transform .18s;
    }
    button:hover { background: #1949b8; transform: translateY(-1px); }
    button:disabled { opacity: .65; cursor: wait; }
    #error {
      margin-top: 16px; padding: 14px 16px; border: 1px solid #f2dfb7;
      border-radius: 10px; background: #fff9ed; color: #80520b;
      font-size: 13px; line-height: 1.55; text-align: left;
    }
    #error[hidden] { display: none; }
    .trust-note { margin: 17px 4px 0; color: #79879a; font-size: 12px; line-height: 1.55; text-align: left; }
    .steps {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;
      width: min(100%, 790px); margin: 32px auto 0;
    }
    .step {
      padding: 19px 18px; border: 1px solid #e6ebf2; border-radius: 13px;
      background: #ffffffb8; text-align: left;
    }
    .step-number {
      width: 27px; height: 27px; display: grid; place-items: center;
      border-radius: 8px; background: #eaf0ff; color: #245bd6;
      font-size: 12px; font-weight: 800;
    }
    .step h2 { margin: 12px 0 5px; font-size: 14px; }
    .step p { margin: 0; color: #718096; font-size: 12px; line-height: 1.55; }
    footer {
      padding: 26px 0 22px; color: #8491a3; font-size: 12px;
      text-align: center;
    }
    @media (max-width: 640px) {
      .topbar { height: 66px; padding: 0 20px; }
      .layout { width: min(100% - 32px, 520px); }
      .hero { padding: 52px 0 24px; }
      .hero-copy { font-size: 15px; }
      .checker-card { padding: 20px; }
      .input-row { flex-direction: column; padding: 8px; }
      input { min-height: 45px; }
      button { min-height: 48px; }
      .steps { grid-template-columns: 1fr; gap: 10px; margin-top: 22px; }
      .step { display: grid; grid-template-columns: 32px 1fr; column-gap: 11px; padding: 14px; }
      .step-number { grid-row: span 2; }
      .step h2 { margin: 1px 0 3px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><span class="brand-mark" aria-hidden="true">S</span><span>SecureSight</span></div>
    <a class="top-link" href="/docs">API documentation</a>
  </header>
  <main class="layout">
    <section class="hero">
      <span class="eyebrow"><span class="eyebrow-dot"></span> URL SECURITY CHECK</span>
      <h1>Check a link before<br>you click.</h1>
      <p class="hero-copy">Get a quick machine-learning assessment of a website address and spot URL patterns that may be suspicious.</p>
      <div class="checker-card">
        <form id="check-form">
          <label for="url">Website address</label>
          <div class="input-row">
            <input id="url" name="url" type="text" placeholder="Paste or type a URL, e.g. https://example.com"
                   autocomplete="url" required>
            <button id="submit" type="submit">Check URL <span aria-hidden="true">→</span></button>
          </div>
        </form>
        <section id="error" role="alert" hidden></section>
        <p class="trust-note">For your privacy, the URL is checked by the local model running on your computer. This tool estimates risk from the URL structure only.</p>
      </div>
      <div class="steps">
        <article class="step">
          <span class="step-number">01</span>
          <h2>Paste a link</h2>
          <p>Enter the website address you want to check.</p>
        </article>
        <article class="step">
          <span class="step-number">02</span>
          <h2>Review the result</h2>
          <p>The model checks URL structure for learned patterns.</p>
        </article>
        <article class="step">
          <span class="step-number">03</span>
          <h2>Stay cautious</h2>
          <p>A result is an estimate, not a guarantee of safety.</p>
        </article>
      </div>
    </section>
  </main>
  <footer>SecureSight · Local URL phishing assessment · Customized by Raj Laxmi Kumari</footer>
  <script>
    const form = document.getElementById("check-form");
    const input = document.getElementById("url");
    const button = document.getElementById("submit");
    const errorMessage = document.getElementById("error");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Analyzing...";
      errorMessage.hidden = true;
      errorMessage.textContent = "";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: input.value })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not check that address.");

        const details = new URLSearchParams({
          url: input.value,
          probability: String(data.probability),
          prediction: String(data.prediction)
        });
        window.location.assign("/details?" + details.toString());
      } catch (error) {
        errorMessage.hidden = false;
        errorMessage.textContent = error.message || "Could not reach the checker. Please try again.";
        errorMessage.scrollIntoView({ behavior: "smooth", block: "center" });
      } finally {
        button.disabled = false;
        button.textContent = "Check URL →";
      }
    });
  </script>
</body>
</html>"""

DETAILS_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>URL Assessment · SecureSight</title>
  <style>
    :root { color-scheme: light; font-family: Inter, "Segoe UI", Arial, sans-serif; color: #13233b; }
    * { box-sizing: border-box; }
    body { min-width: 320px; min-height: 100vh; margin: 0; background: #f4f7fb; }
    header {
      height: 76px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 max(28px, calc((100vw - 900px) / 2));
      background: #fff; border-bottom: 1px solid #e8edf4;
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 750; letter-spacing: -.02em; }
    .brand-mark {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 12px; color: white; background: #245bd6; font-size: 19px;
    }
    header a, .links a { color: #245bd6; font-size: 14px; font-weight: 700; text-decoration: none; }
    header a:hover, .links a:hover { text-decoration: underline; }
    main { width: min(900px, calc(100% - 40px)); margin: 54px auto; }
    .eyebrow {
      color: #245bd6; font-size: 12px; font-weight: 750; letter-spacing: .08em;
      text-transform: uppercase;
    }
    h1 { margin: 13px 0 8px; font-size: clamp(30px, 5vw, 40px); letter-spacing: -.04em; }
    .intro { margin: 0; color: #718096; line-height: 1.6; }
    .details-card {
      margin-top: 28px; padding: 28px; border: 1px solid #e6ebf2;
      border-radius: 16px; background: #fff; box-shadow: 0 18px 48px #14294d0d;
    }
    dl { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 0; }
    .detail-item { min-width: 0; padding: 17px; border: 1px solid #edf0f5; border-radius: 11px; background: #fbfcfe; }
    dt { color: #77859a; font-size: 12px; font-weight: 650; }
    dd { margin: 8px 0 0; overflow-wrap: anywhere; color: #1b2a42; font-size: 15px; font-weight: 700; line-height: 1.5; }
    .result-phishing { color: #b83c32; }
    .result-legitimate { color: #188153; }
    .verdict {
      display: flex; align-items: flex-start; gap: 16px; padding: 22px;
      border: 1px solid #d9e8df; border-radius: 14px; background: #f1faf4;
    }
    .verdict.phishing { border-color: #f0d8d5; background: #fff5f3; }
    .verdict-icon {
      width: 42px; height: 42px; flex: 0 0 42px; display: grid;
      place-items: center; border-radius: 12px; background: #dff3e6;
      color: #16804b; font-size: 21px; font-weight: 800;
    }
    .phishing .verdict-icon { background: #fbe3df; color: #b83c32; }
    .verdict h2 { margin: 1px 0 5px; color: #17663a; font-size: 20px; letter-spacing: -.02em; }
    .phishing h2 { color: #a52c25; }
    .verdict p { margin: 0; color: #64748b; font-size: 13px; line-height: 1.55; }
    .score-label { display: flex; justify-content: space-between; gap: 12px; }
    .score-track { height: 8px; overflow: hidden; margin-top: 12px; border-radius: 99px; background: #e7edf4; }
    .score-fill { width: 0; height: 100%; border-radius: inherit; background: #31a875; transition: width .4s; }
    .score-fill.phishing { background: #d65348; }
    .explanation { margin: 22px 0 0; color: #66758b; font-size: 14px; line-height: 1.7; }
    .warning {
      margin: 17px 0 0; padding: 15px 17px; border: 1px solid #f1dfb8;
      border-radius: 11px; background: #fff9ed; color: #80520b; font-size: 13px; line-height: 1.65;
    }
    .links { display: flex; flex-wrap: wrap; gap: 22px; margin-top: 24px; }
    footer { margin: 30px 0 0; color: #8793a4; font-size: 12px; text-align: center; }
    @media (max-width: 580px) {
      header { height: 66px; padding: 0 20px; }
      main { width: calc(100% - 32px); margin: 36px auto; }
      .details-card { padding: 18px; }
      dl { grid-template-columns: 1fr; gap: 11px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="brand-mark" aria-hidden="true">S</span><span>SecureSight</span></div>
    <a href="/">← Back to checker</a>
  </header>
  <main>
    <span class="eyebrow">ANALYSIS SUMMARY</span>
    <h1>URL assessment</h1>
    <p class="intro">Your check is complete. Review the model's assessment below.</p>
    <section class="details-card">
      <div class="verdict" id="verdict" role="status">
        <span class="verdict-icon" id="verdict-icon" aria-hidden="true">✓</span>
        <div>
          <h2 id="verdict-title">Assessment unavailable</h2>
          <p id="verdict-copy">Return to the checker and submit a URL to get an assessment.</p>
        </div>
      </div>
      <dl>
        <div class="detail-item"><dt>Website checked</dt><dd id="checked-url">Not provided</dd></div>
        <div class="detail-item"><dt>Model result</dt><dd id="model-result">No result found. Go back and check a website first.</dd></div>
        <div class="detail-item">
          <dt class="score-label"><span>Model phishing probability</span><span id="score">Not available</span></dt>
          <div class="score-track" id="score-track" role="progressbar" aria-label="Model phishing probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div class="score-fill" id="score-fill"></div>
          </div>
        </div>
        <div class="detail-item"><dt>What was analyzed</dt><dd>URL structure only</dd></div>
      </dl>
      <p class="explanation">This percentage is the model's estimated probability for its phishing class, based only on URL structure. It is not a calibrated real-world risk measurement, and the model can make mistakes.</p>
      <p class="warning">A “Legitimate” result is not a safety guarantee. Do not enter passwords, payment details, or personal information on a website you are unsure about.</p>
      <div class="links">
        <a href="/">Back to checker</a>
        <a href="/docs">Technical API details</a>
      </div>
    </section>
  </main>
  <footer>SecureSight · Local interface customized by Raj Laxmi Kumari</footer>
  <script>
    const params = new URLSearchParams(window.location.search);
    const url = params.get("url");
    const prediction = params.get("prediction");
    const probability = Number(params.get("probability"));

    if (url) document.getElementById("checked-url").textContent = url;
    if (prediction === "true" || prediction === "false") {
      const result = document.getElementById("model-result");
      result.textContent = prediction === "true"
        ? "Possible phishing signs detected"
        : "No phishing signs detected";
      result.className = prediction === "true" ? "result-phishing" : "result-legitimate";
      const verdict = document.getElementById("verdict");
      const isPhishing = prediction === "true";
      verdict.classList.toggle("phishing", isPhishing);
      document.getElementById("verdict-icon").textContent = isPhishing ? "!" : "✓";
      document.getElementById("verdict-title").textContent = isPhishing
        ? "Potential phishing risk"
        : "No phishing signs detected";
      document.getElementById("verdict-copy").textContent = isPhishing
        ? "The model found URL patterns often associated with phishing. Avoid opening this link."
        : "The model did not find strong phishing patterns in this URL. This is not a guarantee of safety.";
    }
    if (Number.isFinite(probability)) {
      const percentage = Math.max(0, Math.min(100, probability * 100));
      document.getElementById("score").textContent = percentage.toFixed(1) + "%";
      document.getElementById("score-fill").style.width = percentage + "%";
      document.getElementById("score-fill").classList.toggle(
        "phishing",
        prediction === "true"
      );
      document.getElementById("score-track").setAttribute("aria-valuenow", String(percentage));
    }
  </script>
</body>
</html>"""


class PredictionRequest(BaseModel):
    url: str = Field(
        min_length=1,
        description="The complete URL to classify. A missing scheme is treated as http://.",
        examples=["https://www.example.com/login"],
    )


class PredictionResponse(BaseModel):
    probability: float = Field(
        ge=0,
        le=1,
        description="Model-estimated probability for the phishing class (0 to 1).",
    )
    prediction: bool = Field(
        description="True if the saved classifier labels the URL as phishing.",
    )
    result: Literal["Phishing", "Legitimate"] = Field(
        description="Human-readable version of the classifier's prediction.",
    )


def extract_model_features(url: str) -> pd.DataFrame:
    url = url.strip()
    if not url:
        raise ValueError("Enter a URL to check.")
    if not urlsplit(url).scheme:
        url = f"http://{url}"

    parsed_url = urlsplit(url)
    extractor = URLFeaturesExtraction(url)
    features = extractor.extract_features()
    hostname = parsed_url.hostname or ""
    domain_tokens = hostname.replace(".", " ").replace("-", " ").replace("_", " ").split()
    sld = extractor.sld
    sld_tokens = sld.replace(".", " ").replace("-", " ").replace("_", " ").split()

    try:
        ipaddress.IPv4Address(hostname)
        has_ipv4 = 1
    except ipaddress.AddressValueError:
        has_ipv4 = 0

    try:
        has_port = int(parsed_url.port is not None)
    except ValueError as error:
        raise ValueError("The URL contains an invalid port.") from error

    features.update(
        {
            "url_has_ipv4": has_ipv4,
            "url_has_port": has_port,
            "sld_token_count": len(sld_tokens),
            "total_tokens": features["domain_token_count"] + features["path_token_count"],
        }
    )
    return pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


@app.get(
    "/",
    summary="Open the URL checker",
    description="Open SecureSight's main page to enter a URL and request a check.",
    tags=["Browser pages"],
)
def home():
    return HTMLResponse(HOME_PAGE)


@app.get("/docs", include_in_schema=False)
def api_guide():
    return HTMLResponse(API_GUIDE_PAGE)


@app.get("/api-reference", include_in_schema=False)
def interactive_api_reference():
    swagger_page = get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title="SecureSight · Interactive API",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
        },
    )
    page = swagger_page.body.decode("utf-8")
    page = page.replace("</head>", f"{INTERACTIVE_REFERENCE_CSS}</head>", 1)
    page = page.replace(
        "<body>",
        """<body>
  <header class="secure-header">
    <div class="secure-brand"><span class="secure-mark" aria-hidden="true">S</span><span>SecureSight</span></div>
    <nav class="secure-nav" aria-label="Main navigation">
      <a href="/">Main checker</a>
      <a href="/docs">API guide</a>
    </nav>
  </header>""",
        1,
    )
    page = page.replace(
        "</body>",
        '<footer class="reference-credit">SecureSight · Local interface customized by Raj Laxmi Kumari</footer></body>',
        1,
    )
    page = page.replace(
        '<div id="swagger-ui">',
        """<section class="reference-hero">
    <span class="reference-kicker">SECURESIGHT · DEVELOPER TOOLS</span>
    <h1>Interactive API reference</h1>
    <p>Explore the available browser pages and test URL classification requests. Expand an endpoint to view its inputs and responses, then use “Try it out” to send a request to the local model.</p>
    <div class="reference-meta"><span>Local model</span><span>Interactive requests</span><span>OpenAPI 3.1</span></div>
  </section>
  <div id="swagger-ui">""",
        1,
    )
    return HTMLResponse(page)


@app.get(
    "/details",
    summary="View check details",
    description="Review the result and model-estimated phishing probability for a checked URL.",
    tags=["Browser pages"],
)
def details():
    return HTMLResponse(DETAILS_PAGE)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a URL",
    description=(
        "Extracts structural features from the supplied URL and classifies it "
        "with the saved local model. It does not visit or inspect the website. "
        "A legitimate result is not a safety guarantee."
    ),
    tags=["URL classification"],
    responses={
        400: {"description": "The URL contains invalid input, such as an invalid port."},
    },
)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        features = extract_model_features(request.url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    probability = float(model.predict_proba(features)[0][1])
    prediction = bool(model.predict(features)[0])
    return PredictionResponse(
        probability=probability,
        prediction=prediction,
        result="Phishing" if prediction else "Legitimate",
    )
