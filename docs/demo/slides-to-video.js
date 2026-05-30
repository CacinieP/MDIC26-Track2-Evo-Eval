/**
 * Slide-to-Video Generator
 * Uses Puppeteer to render HTML slides as frames, then FFmpeg to combine with TTS audio.
 *
 * Usage: node slides-to-video.js
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const SLIDES_DIR = path.join(__dirname);
const AUDIO_DIR = path.join(__dirname, 'audio');
const OUTPUT_DIR = path.join(__dirname);
const TEMP_DIR = path.join(__dirname, 'temp_frames');

const VIDEO_WIDTH = 1920;
const VIDEO_HEIGHT = 1080;
const FPS = 25;

// Slide durations (seconds) - should match slides.html data-duration
const SLIDE_DURATIONS = [5, 8, 8, 8, 6, 8, 6, 4];

async function main() {
  console.log('=== Slide-to-Video Generator ===\n');

  // Create temp directory
  if (!fs.existsSync(TEMP_DIR)) fs.mkdirSync(TEMP_DIR, { recursive: true });

  // Step 1: Get audio durations
  console.log('Step 1: Measuring audio durations...');
  const audioDurations = [];
  for (let i = 1; i <= 8; i++) {
    const audioFile = path.join(AUDIO_DIR, `slide${i}.mp3`);
    if (!fs.existsSync(audioFile)) {
      console.error(`  Missing: slide${i}.mp3`);
      audioDurations.push(SLIDE_DURATIONS[i - 1]);
      continue;
    }
    try {
      const result = execSync(
        `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${audioFile}"`,
        { encoding: 'utf-8' }
      );
      const dur = parseFloat(result.trim());
      audioDurations.push(Math.ceil(dur) + 1); // +1s buffer
      console.log(`  slide${i}.mp3: ${dur.toFixed(1)}s -> ${Math.ceil(dur) + 1}s`);
    } catch (e) {
      audioDurations.push(SLIDE_DURATIONS[i - 1]);
      console.log(`  slide${i}.mp3: using default ${SLIDE_DURATIONS[i - 1]}s`);
    }
  }

  // Step 2: Use Puppeteer to render each slide as an image
  console.log('\nStep 2: Rendering slides with Puppeteer...');
  let puppeteer;
  try {
    puppeteer = require('puppeteer');
  } catch {
    console.log('  Puppeteer not found, installing...');
    execSync('npm install puppeteer', { cwd: __dirname, stdio: 'inherit' });
    puppeteer = require('puppeteer');
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: VIDEO_WIDTH, height: VIDEO_HEIGHT, deviceScaleFactor: 1 });

  const slidesPath = path.join(SLIDES_DIR, 'slides.html');
  await page.goto(`file:///${slidesPath.replace(/\\/g, '/')}`, { waitUntil: 'networkidle0' });

  for (let i = 0; i < 8; i++) {
    const slideNum = i + 1;
    // Activate this slide
    await page.evaluate((idx) => {
      document.querySelectorAll('.slide').forEach((s, j) => {
        s.classList.toggle('active', j === idx);
      });
    }, i);

    await new Promise(r => setTimeout(r, 500)); // Wait for transition

    const outFile = path.join(TEMP_DIR, `slide${slideNum}.png`);
    await page.screenshot({ path: outFile, type: 'png' });
    console.log(`  Rendered slide${slideNum}.png (${audioDurations[i]}s)`);
  }

  await browser.close();

  // Step 3: Generate video segments for each slide
  console.log('\nStep 3: Generating video segments...');
  const segmentFiles = [];

  for (let i = 0; i < 8; i++) {
    const slideNum = i + 1;
    const imgFile = path.join(TEMP_DIR, `slide${slideNum}.png`);
    const audioFile = path.join(AUDIO_DIR, `slide${slideNum}.mp3`);
    const segFile = path.join(TEMP_DIR, `segment${slideNum}.mp4`);
    const duration = audioDurations[i];

    let cmd;
    if (fs.existsSync(audioFile)) {
      // Image + Audio -> Video segment
      cmd = `ffmpeg -y -loop 1 -i "${imgFile}" -i "${audioFile}" ` +
        `-c:v libx264 -tune stillimage -c:a aac -b:a 192k ` +
        `-pix_fmt yuv420p -shortest -t ${duration} ` +
        `-vf "scale=${VIDEO_WIDTH}:${VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad=${VIDEO_WIDTH}:${VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black" ` +
        `"${segFile}"`;
    } else {
      // Image only -> Video segment (no audio)
      cmd = `ffmpeg -y -loop 1 -i "${imgFile}" ` +
        `-c:v libx264 -pix_fmt yuv420p -t ${duration} ` +
        `-vf "scale=${VIDEO_WIDTH}:${VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad=${VIDEO_WIDTH}:${VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black" ` +
        `-r ${FPS} "${segFile}"`;
    }

    try {
      execSync(cmd, { stdio: 'pipe' });
      segmentFiles.push(segFile);
      console.log(`  segment${slideNum}.mp4 (${duration}s)`);
    } catch (e) {
      console.error(`  Failed: segment${slideNum}: ${e.message.slice(0, 100)}`);
    }
  }

  // Step 4: Concatenate all segments
  console.log('\nStep 4: Concatenating segments...');
  const concatFile = path.join(TEMP_DIR, 'concat.txt');
  const concatContent = segmentFiles.map(f => `file '${f.replace(/\\/g, '/')}'`).join('\n');
  fs.writeFileSync(concatFile, concatContent);

  const outputFile = path.join(OUTPUT_DIR, 'demo-video.mp4');
  try {
    execSync(
      `ffmpeg -y -f concat -safe 0 -i "${concatFile}" -c copy "${outputFile}"`,
      { stdio: 'pipe' }
    );
    const size = fs.statSync(outputFile).size / 1024 / 1024;
    console.log(`\n=== Done! ===`);
    console.log(`Output: ${outputFile}`);
    console.log(`Size: ${size.toFixed(1)} MB`);

    // Get total duration
    const durResult = execSync(
      `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${outputFile}"`,
      { encoding: 'utf-8' }
    );
    console.log(`Duration: ${parseFloat(durResult).toFixed(1)}s`);
  } catch (e) {
    console.error(`Concat failed: ${e.message.slice(0, 200)}`);
  }

  // Cleanup temp
  console.log(`\nTemp files in: ${TEMP_DIR}`);
  console.log('Run `rm -rf temp_frames` to clean up.');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
