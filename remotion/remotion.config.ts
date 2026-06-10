import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
// publicDir is passed per-chapter on the CLI (--public-dir <episode dir>) so
// staticFile('scenes/…') and staticFile('tts/clips/…') resolve per chapter.
