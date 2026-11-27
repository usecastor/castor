import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  cacheBinaryPath,
  downloadAndInstallRelease,
  expectedAsset,
  pickNewest,
  pruneStaleCaches,
  readPackageVersion,
  resolveCachedBinary,
  resolveCommand,
  verifySha256,
} from "../bin/castor.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");

function makeTmpDir(prefix = "castor-shim-test-") {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test("expectedAsset maps supported platforms and rejects unsupported ones", () => {
  assert.deepEqual(expectedAsset("linux", "x64", "1.5.0"), {
    target: "linux-x64",
    archiveName: "castor-v1.5.0-linux-x64.tar.gz",
    sha256Name: "castor-v1.5.0-linux-x64.tar.gz.sha256",
    binaryName: "castor",
    format: "tar.gz",
  });

  assert.deepEqual(expectedAsset("win32", "x64", "1.5.0"), {
    target: "win32-x64",
    archiveName: "castor-v1.5.0-win32-x64.zip",
    sha256Name: "castor-v1.5.0-win32-x64.zip.sha256",
    binaryName: "castor.exe",
    format: "zip",
  });

  assert.equal(expectedAsset("darwin", "arm64", "1.5.0"), null);
  assert.equal(expectedAsset("linux", "arm64", "1.5.0"), null);
});

test("cacheBinaryPath places versioned binaries under <state_dir>/bin/v<version>", () => {
  const stateDir = path.join(os.tmpdir(), "custom-castor-state");
  assert.equal(
    cacheBinaryPath({ version: "1.5.0", stateDir, platform: "win32" }),
    path.join(stateDir, "bin", "v1.5.0", "castor.exe")
  );
  assert.equal(
    cacheBinaryPath({ version: "1.5.0", stateDir, platform: "linux" }),
    path.join(stateDir, "bin", "v1.5.0", "castor")
  );
  assert.equal(
    cacheBinaryPath({
      version: "1.5.0",
      homedir: path.join(os.tmpdir(), "alice"),
      platform: "linux",
    }),
    path.join(os.tmpdir(), "alice", ".castor", "bin", "v1.5.0", "castor")
  );
});

test("pickNewest prefers newest local target/{release,debug} binary", () => {
  const tmp = makeTmpDir();
  try {
    const releaseBin = path.join(tmp, "target", "release", "castor.exe");
    const debugBin = path.join(tmp, "target", "debug", "castor.exe");
    fs.mkdirSync(path.dirname(releaseBin), { recursive: true });
    fs.mkdirSync(path.dirname(debugBin), { recursive: true });

    fs.writeFileSync(releaseBin, "release-old");
    fs.writeFileSync(debugBin, "debug-new");

    const earlier = new Date(Date.now() - 60_000);
    const later = new Date(Date.now() - 1_000);
    fs.utimesSync(releaseBin, earlier, earlier);
    fs.utimesSync(debugBin, later, later);

    assert.equal(pickNewest([releaseBin, debugBin]), debugBin);

    const newest = new Date();
    fs.utimesSync(releaseBin, newest, newest);
    assert.equal(pickNewest([releaseBin, debugBin]), releaseBin);

    assert.equal(pickNewest([path.join(tmp, "missing")]), null);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("resolveCommand prioritizes local target/ binary over cache and download", async () => {
  const tmp = makeTmpDir();
  try {
    const fakeRoot = path.join(tmp, "repo");
    const stateDir = path.join(tmp, "state");
    const localBin = path.join(fakeRoot, "target", "release", "castor.exe");
    const cachedBin = cacheBinaryPath({
      version: "1.5.0",
      stateDir,
      platform: "win32",
    });

    fs.mkdirSync(path.dirname(localBin), { recursive: true });
    fs.mkdirSync(path.dirname(cachedBin), { recursive: true });
    fs.writeFileSync(localBin, "local");
    fs.writeFileSync(cachedBin, "cached");

    let downloadCalled = false;
    const resolved = await resolveCommand({
      root: fakeRoot,
      args: ["--help"],
      env: { CASTOR_STATE_DIR: stateDir },
      platform: "win32",
      arch: "x64",
      version: "1.5.0",
      downloadFn: async () => {
        downloadCalled = true;
        return cachedBin;
      },
    });

    assert.equal(downloadCalled, false);
    assert.equal(resolved.cmd, localBin);
    assert.deepEqual(resolved.args, ["--help"]);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("resolveCommand uses cached binary without downloading when target/ is absent", async () => {
  const tmp = makeTmpDir();
  try {
    const fakeRoot = path.join(tmp, "repo");
    const stateDir = path.join(tmp, "state");
    const cachedBin = cacheBinaryPath({
      version: "1.5.0",
      stateDir,
      platform: "win32",
    });
    fs.mkdirSync(path.dirname(cachedBin), { recursive: true });
    fs.writeFileSync(cachedBin, "cached-exe");

    let downloadCalled = false;
    const resolved = await resolveCommand({
      root: fakeRoot,
      args: ["status"],
      env: { CASTOR_STATE_DIR: stateDir },
      platform: "win32",
      arch: "x64",
      version: "1.5.0",
      downloadFn: async () => {
        downloadCalled = true;
        throw new Error("should not download");
      },
    });

    assert.equal(downloadCalled, false);
    assert.equal(resolved.cmd, cachedBin);
    assert.equal(resolveCachedBinary({ version: "1.5.0", stateDir, platform: "win32" }), cachedBin);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("resolveCommand reports clear error with usecastor/castor URL when no binary or download is available", async () => {
  const tmp = makeTmpDir();
  try {
    await assert.rejects(
      () =>
        resolveCommand({
          root: path.join(tmp, "empty-repo"),
          args: ["--help"],
          env: {
            CASTOR_STATE_DIR: path.join(tmp, "empty-state"),
            CASTOR_NO_DOWNLOAD: "1",
          },
          platform: "win32",
          arch: "x64",
          version: "1.5.0",
        }),
      (err) => {
        assert.match(err.message, /usecastor\/castor/);
        assert.match(err.message, /CASTOR_NO_DOWNLOAD/);
        return true;
      }
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("verifySha256 validates matching checksums and rejects mismatches", async () => {
  const tmp = makeTmpDir();
  try {
    const filePath = path.join(tmp, "castor-test.bin");
    const payload = Buffer.from("castor-release-payload-v1.5.0");
    fs.writeFileSync(filePath, payload);

    const expectedHash = crypto.createHash("sha256").update(payload).digest("hex");
    const shaFileContent = `${expectedHash}  castor-v1.5.0-linux-x64.tar.gz\n`;

    const res = await verifySha256(filePath, shaFileContent);
    assert.equal(res.actual, expectedHash);
    assert.equal(res.expected, expectedHash);

    const badHash = "0".repeat(64);
    await assert.rejects(
      () => verifySha256(filePath, `${badHash}  castor-v1.5.0-linux-x64.tar.gz\n`),
      /SHA-256 checksum mismatch/i
    );

    await assert.rejects(() => verifySha256(filePath, "not-a-hash"), /Invalid SHA-256/);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("pruneStaleCaches removes older v* dirs while keeping current version and non-version dirs", () => {
  const tmp = makeTmpDir();
  try {
    const cacheRoot = path.join(tmp, "bin");
    fs.mkdirSync(path.join(cacheRoot, "v1.3.0"), { recursive: true });
    fs.mkdirSync(path.join(cacheRoot, "v1.4.0"), { recursive: true });
    fs.mkdirSync(path.join(cacheRoot, "v1.5.0"), { recursive: true });
    fs.mkdirSync(path.join(cacheRoot, "custom-notes"), { recursive: true });

    const pruned = pruneStaleCaches(cacheRoot, "1.5.0");
    assert.deepEqual(pruned.sort(), ["v1.3.0", "v1.4.0"]);
    assert.equal(fs.existsSync(path.join(cacheRoot, "v1.3.0")), false);
    assert.equal(fs.existsSync(path.join(cacheRoot, "v1.4.0")), false);
    assert.equal(fs.existsSync(path.join(cacheRoot, "v1.5.0")), true);
    assert.equal(fs.existsSync(path.join(cacheRoot, "custom-notes")), true);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("node bin/castor.js --help executes against the existing local binary", () => {
  const version = readPackageVersion(path.join(REPO_ROOT, "package.json"));
  assert.equal(version, "1.5.0");

  const tmpState = makeTmpDir("castor-help-state-");
  try {
    const res = spawnSync(process.execPath, [path.join(REPO_ROOT, "bin", "castor.js"), "--help"], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      env: {
        ...process.env,
        CASTOR_STATE_DIR: tmpState,
        CASTOR_NO_DOWNLOAD: "1",
      },
    });

    assert.equal(
      res.status,
      0,
      `Expected exit 0 from castor --help, got ${res.status}. stderr:\n${res.stderr}`
    );
    assert.match(res.stdout, /castor/i);
  } finally {
    fs.rmSync(tmpState, { recursive: true, force: true });
  }
});

test("downloadAndInstallRelease verifies SHA-256, extracts archive, caches binary, and prunes stale versions", async () => {
  const tmp = makeTmpDir("castor-dl-test-");
  try {
    const platform = process.platform === "win32" ? "win32" : "linux";
    const asset = expectedAsset(platform, "x64", "1.5.0");
    assert.ok(asset);

    const srcDir = path.join(tmp, "src");
    fs.mkdirSync(srcDir, { recursive: true });
    const srcBin = path.join(srcDir, asset.binaryName);
    fs.writeFileSync(srcBin, "#!/bin/sh\necho fake-castor\n");

    const archivePath = path.join(tmp, asset.archiveName);
    if (asset.format === "zip") {
      const zipRes = spawnSync(
        "powershell.exe",
        [
          "-NoProfile",
          "-NonInteractive",
          "-Command",
          "Compress-Archive -LiteralPath $env:CASTOR_SRC -DestinationPath $env:CASTOR_ZIP -Force",
        ],
        {
          encoding: "utf8",
          windowsHide: true,
          env: { ...process.env, CASTOR_SRC: srcBin, CASTOR_ZIP: archivePath },
        }
      );
      assert.equal(zipRes.status, 0, zipRes.stderr);
    } else {
      const tarRes = spawnSync("tar", ["-czf", archivePath, "-C", srcDir, asset.binaryName], {
        encoding: "utf8",
      });
      assert.equal(tarRes.status, 0, tarRes.stderr);
    }

    const archiveBytes = fs.readFileSync(archivePath);
    const validSha = crypto.createHash("sha256").update(archiveBytes).digest("hex");

    const stateDir = path.join(tmp, "state");
    fs.mkdirSync(path.join(stateDir, "bin", "v1.4.0"), { recursive: true });

    const mockFetch = async (url) => {
      if (url.endsWith(".sha256")) {
        return new Response(`${validSha}  ${asset.archiveName}\n`, { status: 200 });
      }
      if (url.endsWith(asset.archiveName)) {
        return new Response(archiveBytes, { status: 200 });
      }
      return new Response("not found", { status: 404, statusText: "Not Found" });
    };

    const installedPath = await downloadAndInstallRelease({
      version: "1.5.0",
      stateDir,
      platform,
      arch: "x64",
      fetchImpl: mockFetch,
    });

    assert.equal(
      installedPath,
      cacheBinaryPath({ version: "1.5.0", stateDir, platform })
    );
    assert.equal(fs.existsSync(installedPath), true);
    assert.equal(fs.existsSync(path.join(stateDir, "bin", "v1.4.0")), false);

    // Ensure checksum mismatch rejects without overwriting or leaving a valid install
    const badStateDir = path.join(tmp, "bad-state");
    const badFetch = async (url) => {
      if (url.endsWith(".sha256")) {
        return new Response(`${"f".repeat(64)}  ${asset.archiveName}\n`, { status: 200 });
      }
      return new Response(archiveBytes, { status: 200 });
    };
    await assert.rejects(
      () =>
        downloadAndInstallRelease({
          version: "1.5.0",
          stateDir: badStateDir,
          platform,
          arch: "x64",
          fetchImpl: badFetch,
        }),
      /SHA-256 checksum mismatch/i
    );
    assert.equal(
      fs.existsSync(cacheBinaryPath({ version: "1.5.0", stateDir: badStateDir, platform })),
      false
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
