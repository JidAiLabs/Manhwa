// Types for render.plan.json (produced by tools/timeline_planner.py).
// This is the SAME contract tools/blender_vse_from_plan.py consumes — the
// planner stays the single source of truth; renderers are interchangeable.

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

// Parity with the Blender script's caps (its CLI defaults).
export const MAX_ZOOM_CAP = 1.35;
export const TEXT_ZOOM_CAP = 1.06;
export const PAN_CAP_FRAC = 0.1;
export const DEFAULT_SAFE_INSET = 0.06;

export type Bias = {x?: number; y?: number};

export type Motion = {
  mode?: string;
  strength?: number;
  ease?: string;
  start_bias?: Bias;
  end_bias?: Bias;
  zoom?: {start?: number; end?: number};
  bg_fill?: {enabled?: boolean; amount?: number; dim?: number};
  fg_fit?: {mode?: string; safe_inset_pct?: number};
  transition?: {in?: {type: string; dur_sec: number}; out?: {type: string; dur_sec: number}};
};

export type Camera = {
  avoid_text_zoom?: boolean;
  max_zoom?: number;
};

export type Cut = {
  file: string;
  start: number;
  dur: number;
  // render_prep splits over-merged crops: layout "split2" shows file + file2
  // side by side on one screen.
  file2?: string;
  layout?: string;
};

export type TimelineItem = {
  segment_id: string;
  group_id?: number;
  start_sec: number;
  duration_sec: number;
  end_sec: number;
  tts_audio?: string;
  cuts?: Cut[];
  scene_files?: string[];
  motion?: Motion;
  camera?: Camera;
  // "intro" | "outro" — inserted by tools/render_prep.py; the renderer
  // supplies the bundled channel audio/visuals for these items.
  branding?: string;
};

export type SceneDims = {
  w: number;
  h: number;
  // document/UI panel (render_prep): never cover-crop or scroll its text
  doc?: boolean;
};

export type RenderPlan = {
  timeline: TimelineItem[];
  total_duration_sec?: number;
  // written by tools/render_prep.py: cleaned/trimmed copies + their sizes
  scenes_subdir?: string;
  scene_dims?: Record<string, SceneDims>;
};

// Panels at least this wide (w/h) render full-bleed (cover) instead of
// contained-with-margins — the page-margin look the user rejected.
export const WIDE_COVER_MIN_ASPECT = 1.3;

// Panels at least this tall (h/w) get a SCROLL SHOT: full-width display with
// the camera travelling vertically — contain-fit renders them unreadably
// small (the user's CRASH/ROAR strips).
export const TALL_SCROLL_MIN_ASPECT = 2.0;

export const toFrames = (sec: number): number => Math.max(1, Math.ceil(sec * FPS));
export const toStartFrame = (sec: number): number => Math.round(sec * FPS);

// tts_audio in the plan is an absolute local path; staticFile() needs a path
// relative to --public-dir (the episode dir), which always contains tts/.
export const publicRelAudio = (ttsAudio: string): string => {
  const i = ttsAudio.indexOf('/tts/');
  return i >= 0 ? ttsAudio.slice(i + 1) : ttsAudio;
};

export const zoomCap = (camera?: Camera): number => {
  const planMax = camera?.max_zoom ?? MAX_ZOOM_CAP;
  const hard = Math.min(MAX_ZOOM_CAP, planMax);
  return camera?.avoid_text_zoom ? Math.min(hard, TEXT_ZOOM_CAP) : hard;
};

export const clamp = (x: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, x));
