#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_URL = "https://github.com/usecastor/castor";
const RELEASES_BASE_URL = `${REPO_URL}/releases/download`;
const DOWNLOAD_TIMEOUT_MS = 60_000;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(__dirname, "..");

export function readPackageVersion(pkgPath = path.join(DEFAULT_ROOT, "package.json")) {
  const raw = fs.readFileSync(pkgPath, "utf8");
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed.version !== "string" || !parsed.version.trim()) {
    throw new Error(`Invalid version in ${pkgPath}`);
  }
  return parsed.version.trim();
}

export function isTruthyEnv(value) {
  if (value === undefined || value === null) return false;
  const normalized = String(value).trim().toLowerCase();
  return normalized !== "" && normalized !== "0" && normalized !== "false" && normalized !== "no";
}

export function pickNewest(paths) {
  let newest = null;
  let maxTime = -1;
  for (const p of paths) {
    try {
      const stat = fs.statSync(p);
      if (stat.isFile() && stat.mtimeMs > maxTime) {
        maxTime = stat.mtimeMs;
        newest = p;
      }
    } catch {
      // Ignore missing or inaccessible candidates.
    }
  }
  return newest;
}

export function expectedAsset(platform, arch, version) {
  if (arch !== "x64") return null;
  if (platform === "linux") {
    const archiveName = `castor-v${version}-linux-x64.tar.gz`;
    return {
      target: "linux-x64",
      archiveName,
      sha256Name: `${archiveName}.sha256`,
      binaryName: "castor",
      format: "tar.gz",
    };
  }
  if (platform === "win32") {
    const archiveName = `castor-v${version}-win32-x64.zip`;
    return {
      target: "win32-x64",
      archiveName,
      sha256Name: `${archiveName}.sha256`,
      binaryName: "castor.exe",
      format: "zip",
    };
  }
  return null;
}

export function resolveCacheRoot(env = process.env, homedir = os.homedir()) {
  const stateDir = env.CASTOR_STATE_DIR || path.join(homedir, ".castor");
  return path.join(stateDir, "bin");
}

export function cacheBinaryPath({
  version,
  stateDir,
  cacheRoot,
  platform = process.platform,
  homedir = os.homedir(),
}) {
  const rootDir =
    cacheRoot ||
    (stateDir ? path.join(stateDir, "bin") : resolveCacheRoot(process.env, homedir));
  const binaryName = platform === "win32" ? "castor.exe" : "castor";
  return path.join(rootDir, `v${version}`, binaryName);
}

export function parseSha256(content) {
  const match = String(content).trim().match(/^([a-fA-F0-9]{64})(?:\s|$)/);
  if (!match) {
    throw new Error("Invalid SHA-256 checksum format");
  }
  return match[1].toLowerCase();
}

export async function verifySha256(filePath, expectedDigestOrContent) {
  const expected = parseSha256(expectedDigestOrContent);
  const actual = await new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex").toLowerCase()));
  });

  if (actual !== expected) {
    throw new Error(
      `SHA-256 checksum mismatch for ${path.basename(filePath)}: expected ${expected}, got ${actual}`
    );
  }
  return { actual, expected };
}

export function pruneStaleCaches(cacheRoot, currentVersion) {
  const keepDir = `v${currentVersion}`;
  const pruned = [];
  let entries;
  try {
    entries = fs.readdirSync(cacheRoot, { withFileTypes: true });
  } catch {
    return pruned;
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (!/^v\d/.test(entry.name)) continue;
    if (entry.name === keepDir) continue;
    fs.rmSync(path.join(cacheRoot, entry.name), { recursive: true, force: true });
    pruned.push(entry.name);
  }
  return pruned;
}

export function isExecutable(filePath, platform = process.platform) {
  try {
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) return false;
    if (platform !== "win32") {
      fs.accessSync(filePath, fs.constants.X_OK);
    }
    return true;
  } catch {
    return false;
  }
}

export function resolveCachedBinary({
  version,
  stateDir,
  cacheRoot,
  platform = process.platform,
  homedir = os.homedir(),
}) {
  const binPath = cacheBinaryPath({ version, stateDir, cacheRoot, platform, homedir });
  return isExecutable(binPath, platform) ? binPath : null;
}

export function resolveLocalDevBinary({ root = DEFAULT_ROOT, platform = process.platform } = {}) {
  const binName = platform === "win32" ? "castor.exe" : "castor";
  return pickNewest([
    path.join(root, "target", "release", binName),
    path.join(root, "target", "debug", binName),
  ]);
}

export function toWslPath(winPath) {
  return winPath
    .replace(/^([a-zA-Z]):/, (_, drive) => `/mnt/${drive.toLowerCase()}`)
    .replace(/\\/g, "/");
}

export function findOnPath(binName, pathEnv = process.env.PATH, platform = process.platform) {
  if (!pathEnv) return null;
  const dirs = pathEnv.split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    const candidate = path.join(dir, binName);
    if (isExecutable(candidate, platform)) {
      return candidate;
    }
  }
  return null;
}

function extractArchive(archivePath, extractDir, format) {
  if (format === "tar.gz") {
    const res = spawnSync("tar", ["-xzf", archivePath, "-C", extractDir], {
      encoding: "utf8",
      windowsHide: true,
    });
    if (res.status !== 0) {
      throw new Error(`Failed to extract ${path.basename(archivePath)}: ${res.stderr || res.error?.message || "tar failed"}`);
    }
    return;
  }

  if (format === "zip") {
    const tarRes = spawnSync("tar", ["-xf", archivePath, "-C", extractDir], {
      encoding: "utf8",
      windowsHide: true,
    });
    if (tarRes.status === 0) return;

    const psRes = spawnSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Expand-Archive -LiteralPath $env:CASTOR_ARCHIVE -DestinationPath $env:CASTOR_DEST -Force",
      ],
      {
        encoding: "utf8",
        windowsHide: true,
        env: {
          ...process.env,
          CASTOR_ARCHIVE: archivePath,
          CASTOR_DEST: extractDir,
        },
      }
    );
    if (psRes.status !== 0) {
      throw new Error(
        `Failed to extract ${path.basename(archivePath)}: ${psRes.stderr || tarRes.stderr || psRes.error?.message || "zip extraction failed"}`
      );
    }
    return;
  }

  throw new Error(`Unsupported archive format: ${format}`);
}

export async function downloadAndInstallRelease({
  version,
  stateDir,
  cacheRoot,
  platform = process.platform,
  arch = process.arch,
  homedir = os.homedir(),
  baseUrl,
  fetchImpl = globalThis.fetch,
}) {
  const asset = expectedAsset(platform, arch, version);
  if (!asset) {
    throw new Error(`No prebuilt Castor binary available for ${platform}-${arch}`);
  }
  if (typeof fetchImpl !== "function") {
    throw new Error("Global fetch() is unavailable; Node.js >= 22 is required");
  }

  const releaseBase = baseUrl || `${RELEASES_BASE_URL}/v${version}`;
  const archiveUrl = `${releaseBase}/${asset.archiveName}`;
  const shaUrl = `${releaseBase}/${asset.sha256Name}`;

  const resolvedCacheRoot =
    cacheRoot ||
    (stateDir ? path.join(stateDir, "bin") : resolveCacheRoot(process.env, homedir));
  const destPath = cacheBinaryPath({
    version,
    cacheRoot: resolvedCacheRoot,
    platform,
    homedir,
  });

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "castor-download-"));
  try {
    const shaResp = await fetchImpl(shaUrl, {
      redirect: "follow",
      signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
    });
    if (!shaResp.ok) {
      throw new Error(`Failed to fetch checksum (${shaResp.status} ${shaResp.statusText}) from ${shaUrl}`);
    }
    const shaText = await shaResp.text();

    const archiveResp = await fetchImpl(archiveUrl, {
      redirect: "follow",
      signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
    });
    if (!archiveResp.ok) {
      throw new Error(`Failed to fetch archive (${archiveResp.status} ${archiveResp.statusText}) from ${archiveUrl}`);
    }

    const archiveBytes = Buffer.from(await archiveResp.arrayBuffer());
    const archivePath = path.join(tmpDir, asset.archiveName);
    fs.writeFileSync(archivePath, archiveBytes);

    await verifySha256(archivePath, shaText);

    const extractDir = path.join(tmpDir, "extract");
    fs.mkdirSync(extractDir, { recursive: true });
    extractArchive(archivePath, extractDir, asset.format);

    const extractedBin = path.join(extractDir, asset.binaryName);
    if (!fs.existsSync(extractedBin)) {
      throw new Error(`Extracted archive did not contain expected binary '${asset.binaryName}'`);
    }

    const destDir = path.dirname(destPath);
    fs.mkdirSync(destDir, { recursive: true });
    const stagedPath = path.join(destDir, `.${asset.binaryName}.tmp-${process.pid}`);
    fs.copyFileSync(extractedBin, stagedPath);
    if (platform !== "win32") {
      fs.chmodSync(stagedPath, 0o755);
    }
    fs.renameSync(stagedPath, destPath);

    pruneStaleCaches(resolvedCacheRoot, version);
    return destPath;
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function buildWslFallback({ root, args }) {
  const linuxBin = pickNewest([
    path.join(root, "target", "release", "castor"),
    path.join(root, "target", "debug", "castor"),
  ]);
  if (!linuxBin) return null;
  const wslBin = toWslPath(linuxBin);
  return {
    cmd: "wsl.exe",
    args: ["--", wslBin, ...args],
    spawnOptions: { stdio: "inherit", windowsHide: true },
  };
}

export async function resolveCommand({
  root = DEFAULT_ROOT,
  args = [],
  env = process.env,
  platform = process.platform,
  arch = process.arch,
  homedir = os.homedir(),
  version = readPackageVersion(path.join(root, "package.json")),
  downloadFn = downloadAndInstallRelease,
} = {}) {
  // 1. Explicit WSL opt-in on Windows.
  if (platform === "win32" && isTruthyEnv(env.CASTOR_USE_WSL)) {
    const wslFallback = buildWslFallback({ root, args });
    if (wslFallback) return wslFallback;
    return {
      cmd: "wsl.exe",
      args: ["--", "castor", ...args],
      spawnOptions: { stdio: "inherit", windowsHide: true },
    };
  }

  // 2. Local dev target/{release,debug} binary (zero network calls).
  const localDev = resolveLocalDevBinary({ root, platform });
  if (localDev) {
    return {
      cmd: localDev,
      args,
      spawnOptions: { stdio: "inherit" },
    };
  }

  // 3. Cached release binary in <state_dir>/bin/v<version>/.
  const cached = resolveCachedBinary({
    version,
    stateDir: env.CASTOR_STATE_DIR,
    platform,
    homedir,
  });
  if (cached) {
    return {
      cmd: cached,
      args,
      spawnOptions: { stdio: "inherit" },
    };
  }

  // 4. Auto-download verified release binary unless CASTOR_NO_DOWNLOAD is set.
  let downloadError = null;
  if (!isTruthyEnv(env.CASTOR_NO_DOWNLOAD)) {
    try {
      const installed = await downloadFn({
        version,
        stateDir: env.CASTOR_STATE_DIR,
        platform,
        arch,
        homedir,
      });
      return {
        cmd: installed,
        args,
        spawnOptions: { stdio: "inherit" },
      };
    } catch (err) {
      downloadError = err;
    }
  }

  // 5. Windows fallback ONLY when a local WSL build already exists in target/.
  if (platform === "win32") {
    const wslFallback = buildWslFallback({ root, args });
    if (wslFallback) {
      return wslFallback;
    }
  } else {
    // 6. POSIX fallback to `castor` installed on PATH (e.g. `cargo install --path .`).
    const pathBin = findOnPath("castor", env.PATH, platform);
    if (pathBin) {
      return {
        cmd: pathBin,
        args,
        spawnOptions: { stdio: "inherit" },
      };
    }
  }

  const reason = downloadError
    ? `Auto-download failed: ${downloadError.message}\n`
    : isTruthyEnv(env.CASTOR_NO_DOWNLOAD)
      ? "Auto-download disabled via CASTOR_NO_DOWNLOAD.\n"
      : "";
  throw new Error(
    `Castor binary not found.\n${reason}\n` +
      `To build or install Castor:\n` +
      `  - Build from source: cargo build --release\n` +
      `  - Install to PATH:   cargo install --path .\n` +
      `  - Releases & Docs:   ${REPO_URL}\n`
  );
}

function handleSpawn(child) {
  child.on("error", (err) => {
    if (err.code === "ENOENT") {
      console.error(
        `\x1b[31m[Castor Error]\x1b[0m Castor binary not found.\n\n` +
          `To build or install Castor:\n` +
          `  - Build from source: cargo build --release\n` +
          `  - Install to PATH:   cargo install --path .\n` +
          `  - Releases & Docs:   ${REPO_URL}\n`
      );
      process.exit(1);
    }
    console.error(err);
    process.exit(1);
  });
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    process.exit(code ?? 1);
  });
}

async function main() {
  try {
    const resolved = await resolveCommand({
      root: DEFAULT_ROOT,
      args: process.argv.slice(2),
    });
    handleSpawn(spawn(resolved.cmd, resolved.args, resolved.spawnOptions));
  } catch (err) {
    console.error(`\x1b[31m[Castor Error]\x1b[0m ${err.message}`);
    process.exit(1);
  }
}

const isMain = (() => {
  if (!process.argv[1]) return false;
  try {
    return fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
  }
})();

if (isMain) {
  main();
}
