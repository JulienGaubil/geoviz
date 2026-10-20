import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';

const payload = JSON.parse(document.getElementById('viewer-payload').textContent);
const loadingOverlay = document.getElementById('viewer-loading');
function setLoadingStatus(message, { hidden = false, error = false } = {}) {
  if (!loadingOverlay) return;
  loadingOverlay.textContent = message;
  loadingOverlay.classList.toggle('hidden', hidden);
  loadingOverlay.classList.toggle('error', error);
}
setLoadingStatus('Loading visible geometry…');
const colormapPresets = payload.colormap_presets || {};
const availableCmaps = payload.available_cmaps || Object.keys(colormapPresets);
const arrayCtors = {
  float32: Float32Array,
  uint8: Uint8Array,
  uint32: Uint32Array,
  int32: Int32Array,
};

function decodeArraySpec(spec, fallbackCtor = Float32Array) {
  if (spec === null || spec === undefined) return null;
  if (ArrayBuffer.isView(spec)) return spec;
  if (!Array.isArray(spec)) {
    return null;
  }
  if (spec.length === 0) {
    return new fallbackCtor(0);
  }
  return new fallbackCtor(Array.isArray(spec[0]) ? spec.flat() : spec);
}

async function hydrateArraySpec(spec) {
  if (spec && spec.encoding === 'raw-url' && typeof spec.url === 'string') {
    const response = await fetch(spec.url);
    if (!response.ok) {
      throw new Error(`Failed to fetch geometry buffer ${spec.url}: HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const expectedBytes = Number(spec.nbytes) || 0;
    if (buffer.byteLength !== expectedBytes) {
      throw new Error(
        `Geometry buffer ${spec.url} has ${buffer.byteLength}/${expectedBytes} bytes`,
      );
    }
    const ctor = arrayCtors[spec.dtype] || Float32Array;
    return new ctor(buffer);
  }
  if (!spec || spec.encoding !== 'gzip-base64-chunks' || !Array.isArray(spec.chunks)) return spec;
  const byteLength = Number(spec.nbytes) || 0;
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (let chunkIdx = 0; chunkIdx < spec.chunks.length; chunkIdx += 1) {
    try {
      const encoded = spec.chunks[chunkIdx];
      const binary = atob(encoded);
      const compressed = new Uint8Array(binary.length);
      for (let idx = 0; idx < binary.length; idx += 1) {
        compressed[idx] = binary.charCodeAt(idx);
      }
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
      const chunk = new Uint8Array(await new Response(stream).arrayBuffer());
      if (offset + chunk.length > bytes.length) {
        throw new Error('decoded data exceeded its declared size');
      }
      bytes.set(chunk, offset);
      offset += chunk.length;
    } catch (error) {
      throw new Error(
        `Failed to decode geometry chunk ${chunkIdx + 1}/${spec.chunks.length}: ${error}`,
      );
    }
  }
  if (offset !== bytes.length) {
    throw new Error(`Geometry buffer was truncated: received ${offset}/${bytes.length} bytes`);
  }
  const ctor = arrayCtors[spec.dtype] || Float32Array;
  return new ctor(bytes.buffer);
}

async function hydrateLayerArrays(layer) {
  const directKeys = [
    'vertices', 'faces', 'face_edges', 'points', 'edges',
    'colors', 'default_colors', 'raw_colors', 'values',
  ];
  await Promise.all(directKeys.map(async (key) => {
    layer[key] = await hydrateArraySpec(layer[key]);
  }));
  if (Array.isArray(layer.edge_chunks)) {
    const hydratedChunks = [];
    for (const chunk of layer.edge_chunks) {
      hydratedChunks.push(await hydrateArraySpec(chunk));
    }
    layer.edge_chunks = hydratedChunks;
  }
  for (const key of [
    'field_colors', 'field_values',
    'edge_field_colors', 'edge_field_values',
    'face_field_colors', 'face_field_values',
  ]) {
    if (Array.isArray(layer[key])) {
      layer[key] = await Promise.all(layer[key].map(hydrateArraySpec));
    }
  }
  if (layer.texture && typeof layer.texture === 'object') {
    layer.texture.uv = await hydrateArraySpec(layer.texture.uv);
  }
}

function normalizeLayerData(layer) {
  const kind = String(layer.layer_kind || (layer.points ? 'pointcloud' : 'mesh')).toLowerCase();
  layer.layer_kind = (kind === 'pointcloud') ? 'pointcloud' : 'mesh';
  layer.vertices = decodeArraySpec(layer.vertices, Float32Array) || new Float32Array(0);
  layer.faces = decodeArraySpec(layer.faces, Uint32Array) || new Uint32Array(0);
  layer.face_size = Number.isFinite(Number(layer.face_size))
    ? Math.max(3, Math.min(4, Math.floor(Number(layer.face_size))))
    : 3;
  layer.points = decodeArraySpec(layer.points, Float32Array) || new Float32Array(0);
  layer.edges = decodeArraySpec(layer.edges, Uint32Array);
  layer.edge_chunks = Array.isArray(layer.edge_chunks)
    ? layer.edge_chunks.map((chunk) => decodeArraySpec(chunk, Uint32Array) || new Uint32Array(0))
    : null;
  layer.face_edges = decodeArraySpec(layer.face_edges, Uint32Array);
  layer.colors = decodeArraySpec(layer.colors, Uint8Array) || new Uint8Array(0);
  layer.default_colors = decodeArraySpec(layer.default_colors, Uint8Array);
  layer.raw_colors = decodeArraySpec(layer.raw_colors, Uint8Array) || layer.colors;
  layer.values = decodeArraySpec(layer.values, Float32Array);
  layer.field_colors = Array.isArray(layer.field_colors)
    ? layer.field_colors.map((colors) => decodeArraySpec(colors, Uint8Array) || new Uint8Array(0))
    : null;
  layer.field_values = Array.isArray(layer.field_values)
    ? layer.field_values.map((values) => decodeArraySpec(values, Float32Array))
    : null;
  layer.edge_field_colors = Array.isArray(layer.edge_field_colors)
    ? layer.edge_field_colors.map((colors) => decodeArraySpec(colors, Uint8Array) || new Uint8Array(0))
    : null;
  layer.edge_field_values = Array.isArray(layer.edge_field_values)
    ? layer.edge_field_values.map((values) => decodeArraySpec(values, Float32Array))
    : null;
  layer.face_field_colors = Array.isArray(layer.face_field_colors)
    ? layer.face_field_colors.map((colors) => decodeArraySpec(colors, Uint8Array) || new Uint8Array(0))
    : null;
  layer.face_field_values = Array.isArray(layer.face_field_values)
    ? layer.face_field_values.map((values) => decodeArraySpec(values, Float32Array))
    : null;
  if (layer.texture && typeof layer.texture === 'object') {
    layer.texture.uv = decodeArraySpec(layer.texture.uv, Float32Array) || new Float32Array(0);
  } else {
    layer.texture = null;
  }
  const fallbackVertexCount = layer.layer_kind === 'pointcloud'
    ? Math.floor(layer.points.length / 3)
    : Math.floor(layer.vertices.length / 3);
  layer.vertex_count = Number.isFinite(layer.vertex_count)
    ? Number(layer.vertex_count)
    : fallbackVertexCount;
  layer.face_count = Number.isFinite(layer.face_count)
    ? Number(layer.face_count)
    : Math.floor(layer.faces.length / layer.face_size);
  layer.point_size = Number.isFinite(Number(layer.point_size))
    ? Number(layer.point_size)
    : Number(payload.point_size);
  layer.edge_thickness = Number.isFinite(Number(layer.edge_thickness))
    ? Number(layer.edge_thickness)
    : 1.0;
  return layer;
}

function createStoredColorAttribute(colors) {
  const attr = new THREE.BufferAttribute(colors, 3, true);
  attr.setUsage(THREE.DynamicDrawUsage);
  return attr;
}

function setStoredColorAttribute(geometry, colors) {
  const colorArray = decodeArraySpec(colors, Uint8Array) || new Uint8Array(0);
  const existing = geometry.getAttribute('color');
  if (
    existing
    && existing.array instanceof Uint8Array
    && existing.array.length === colorArray.length
    && existing.normalized
  ) {
    existing.array.set(colorArray);
    existing.needsUpdate = true;
    return existing;
  }
  const attr = createStoredColorAttribute(colorArray);
  geometry.setAttribute('color', attr);
  return attr;
}

const canvasContainer = document.getElementById('canvas-container');

function getViewerViewportSize() {
  const rect = canvasContainer.getBoundingClientRect();
  return {
    width: Math.max(1, Math.round(rect.width || canvasContainer.clientWidth || window.innerWidth)),
    height: Math.max(1, Math.round(rect.height || canvasContainer.clientHeight || window.innerHeight)),
  };
}

const initialViewport = getViewerViewportSize();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  initialViewport.width / initialViewport.height,
  0.001,
  1e7,
);
camera.position.set(0, 0, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.setSize(initialViewport.width, initialViewport.height, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x111827, 1.0);
canvasContainer.appendChild(renderer.domElement);

// Layer hydration can select an initial scalar field and request a render.
// Initialize this state before visible layers are loaded below.
let renderQueued = false;
let filterApplyQueued = false;

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 2.0;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.dynamicDampingFactor = 0.15;
controls.staticMoving = true;
// Keyboard A/S/D mode overrides can get stuck when focus leaves the canvas.
// Keep gesture selection explicit: primary drag rotates, secondary drag pans.
controls.keys = [];

// Trackball zoom normally shrinks the camera-target distance to zero. Keep
// that orbit radius instead: the camera's dolly also translates its target.
// Wrapping update preserves the stock wheel, middle-drag and pinch handling,
// as well as rotation/panning. A new camera fit naturally establishes a new
// radius because we measure it before each update, not once at startup.
const trackballUpdate = controls.update.bind(controls);
const dollyEye = new THREE.Vector3();
// Keep the orbit anchor in world space, independently of the zoom target.
let surfaceOrbitPivot = null;
let surfaceOrbitDragging = false;
const orbitOldPosition = new THREE.Vector3();
const orbitOldQuaternion = new THREE.Quaternion();
const orbitDelta = new THREE.Quaternion();
const orbitCorrection = new THREE.Vector3();
controls.update = function updateWithTranslatingZoomTarget() {
  const previousRadius = camera.position.distanceTo(controls.target);
  orbitOldPosition.copy(camera.position);
  orbitOldQuaternion.copy(camera.quaternion);
  trackballUpdate();
  if (surfaceOrbitDragging && surfaceOrbitPivot) {
    // Trackball rotates about its look-at target. Translate that rotation to
    // the picked point without snapping the view toward an off-centre click.
    orbitDelta.copy(camera.quaternion).multiply(orbitOldQuaternion.invert());
    orbitCorrection.copy(orbitOldPosition).sub(surfaceOrbitPivot)
      .applyQuaternion(orbitDelta).add(surfaceOrbitPivot).sub(camera.position);
    camera.position.add(orbitCorrection);
    controls.target.add(orbitCorrection);
  }
  dollyEye.subVectors(camera.position, controls.target);
  const nextRadius = dollyEye.length();
  if (
    camera.isPerspectiveCamera && nextRadius > 0 && previousRadius > 0
    && Math.abs(nextRadius - previousRadius) > previousRadius * 1e-10
  ) {
    controls.target.addScaledVector(dollyEye, (nextRadius - previousRadius) / nextRadius);
  }
};

let viewerWidth = initialViewport.width;
let viewerHeight = initialViewport.height;
function syncViewerSize({ closeMenus = false } = {}) {
  const { width, height } = getViewerViewportSize();
  if (closeMenus) closeAllDropdownMenus();
  if (width === viewerWidth && height === viewerHeight) return false;
  viewerWidth = width;
  viewerHeight = height;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  refreshLineMaterialResolutions();
  controls.handleResize?.();
  requestRender();
  return true;
}

// Register before geometry hydration. Large layers may take long enough for a
// browser window or notebook iframe to finish laying out while arrays decode.
window.addEventListener('resize', () => {
  syncViewerSize({ closeMenus: true });
});
const viewerResizeObserver = typeof ResizeObserver === 'undefined'
  ? null
  : new ResizeObserver(() => {
    syncViewerSize();
  });
viewerResizeObserver?.observe(canvasContainer);

const layers = [];
let isDayMode = false;
let commandsVisible = true;
const CLIP_SLIDER_MAX = 1000;
const dropdownPortal = document.getElementById('dropdown-portal');
const filterPanel = document.getElementById('filter-panel');
const filterRulesList = document.getElementById('filter-rules-list');
const filterMaskedOpacityInput = document.getElementById('filter-masked-opacity');
const filterMaskedOpacityValue = document.getElementById('filter-masked-opacity-value');
const filterAddRuleButton = document.getElementById('btn-filter-add-rule');
const filterResetRulesButton = document.getElementById('btn-filter-reset-rules');
const filterToggleButton = document.getElementById('btn-filter-panel');
const filterCloseButton = document.getElementById('btn-filter-close');
const controlsCloseButton = document.getElementById('btn-controls-close');
const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
const hemisphereLight = new THREE.HemisphereLight(0xe0f2fe, 0x111827, 0.7);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
keyLight.position.set(2.5, 3.5, 4.0);
const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.4);
rimLight.position.set(-3.0, 1.5, -2.5);
scene.add(ambientLight, hemisphereLight, keyLight, rimLight);
let filterRuleCounter = 0;
const FILTER_SLIDER_MAX = 1000;
const FILTER_POSITION = 'position';
const FILTER_SCOPE_ALL_KEY = '__all__';
const MAX_MESH_INTERACTION_PROXY_POINTS = 10000;
const MESH_INTERACTION_PROXY_FACE_THRESHOLD = 1000000;
const MESH_EDGE_CHUNK_FACE_COUNT = 500000;
const filterState = {
  open: false,
  maskedOpacity: 0.0,
  rules: [],
};
const pointVertexShader = `
  attribute vec3 color;
  varying vec3 vColor;
  uniform float size;
  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size;
    gl_Position = projectionMatrix * mvPosition;
  }
`;
const pointFragmentShader = `
  varying vec3 vColor;
  uniform float opacity;
  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    if (length(center) > 0.5) discard;
    gl_FragColor = vec4(vColor, opacity);
  }
`;


function clampUnit(value, fallback = 0.0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(1.0, Math.max(0.0, numeric));
}

function createClipState(minFrac = 0.0, maxFrac = 1.0) {
  const start = clampUnit(minFrac, 0.0);
  const end = clampUnit(maxFrac, 1.0);
  return {
    minFrac: Math.min(start, end),
    maxFrac: Math.max(start, end),
  };
}

function getEntryFieldCount(entry, association = 'vertex') {
  const data = getAssociationFieldData(entry, association);
  return Math.max(
    Array.isArray(data.names) ? data.names.length : 0,
    Array.isArray(data.values) ? data.values.length : 0,
    Array.isArray(data.metas) ? data.metas.length : 0,
    Array.isArray(data.colors) ? data.colors.length : 0,
    association === 'vertex' ? 1 : 0,
  );
}

function hasEntryFields(entry, association = 'vertex') {
  const data = getAssociationFieldData(entry, association);
  const hasArrayData = [data.names, data.values, data.metas, data.colors]
    .some((items) => Array.isArray(items) && items.length > 0);
  if (hasArrayData) return true;
  return association === 'vertex' && Boolean(entry.layer.values || entry.layer.scalar_meta);
}

function ensureFieldClipStates(entry, association = 'vertex') {
  const fieldCount = getEntryFieldCount(entry, association);
  const state = getAssociationState(entry, association);
  if (!Array.isArray(state.clipStates)) state.clipStates = [];
  while (state.clipStates.length < fieldCount) {
    state.clipStates.push(createClipState());
  }
  return state.clipStates;
}

function getClipState(entry, fieldIdx = null, association = 'vertex') {
  const associationState = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? associationState.activeFieldIdx ?? 0;
  if (entry.shareFieldScale) {
    if (!associationState.sharedClip) associationState.sharedClip = createClipState();
    return associationState.sharedClip;
  }
  const states = ensureFieldClipStates(entry, association);
  const clampedIdx = Math.max(0, Math.min(states.length - 1, resolvedIdx));
  if (!states[clampedIdx]) {
    states[clampedIdx] = createClipState();
  }
  return states[clampedIdx];
}

function sliderValueToClip(value, fallback = 0.0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return clampUnit(numeric / CLIP_SLIDER_MAX, fallback);
}

function clipToSliderValue(value, fallback = 0.0) {
  return Math.round(clampUnit(value, fallback) * CLIP_SLIDER_MAX);
}

function setClipBoundary(state, boundary, nextFrac) {
  if (!state) return;
  const clamped = clampUnit(nextFrac, boundary === 'max' ? 1.0 : 0.0);
  if (boundary === 'max') {
    state.maxFrac = Math.max(clamped, state.minFrac);
  } else {
    state.minFrac = Math.min(clamped, state.maxFrac);
  }
}

function applyClipToRange(range, clipState) {
  if (!range) return null;
  const minVal = Number(range.min);
  const maxVal = Number(range.max);
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  const span = maxVal - minVal;
  if (!(Math.abs(span) > 1e-12)) {
    return { min: minVal, max: maxVal };
  }
  const startFrac = clampUnit(clipState && clipState.minFrac, 0.0);
  const endFrac = clampUnit(clipState && clipState.maxFrac, 1.0);
  return {
    min: minVal + span * Math.min(startFrac, endFrac),
    max: minVal + span * Math.max(startFrac, endFrac),
  };
}

function formatScalarValue(value) {
  return Number.isFinite(value) ? Number(value).toFixed(4) : 'n/a';
}

function syncAllColorbarClipControls() {
  document.querySelectorAll('.colorbar').forEach((node) => {
    if (typeof node.__syncClipControl === 'function') {
      node.__syncClipControl();
    }
  });
}

function edgeColorHex() {
  return isDayMode ? 0x0f172a : 0xffffff;
}

function applyLightingMode(dayMode) {
  ambientLight.intensity = dayMode ? 0.7 : 0.5;
  hemisphereLight.intensity = dayMode ? 0.85 : 0.65;
  hemisphereLight.color.set(dayMode ? 0xf8fafc : 0xe0f2fe);
  hemisphereLight.groundColor.set(dayMode ? 0xcbd5e1 : 0x111827);
  keyLight.intensity = dayMode ? 1.05 : 1.2;
  keyLight.color.set(dayMode ? 0xffffff : 0xf8fafc);
  rimLight.intensity = dayMode ? 0.28 : 0.42;
}

function applyThemeMode(dayMode) {
  isDayMode = Boolean(dayMode);
  document.body.classList.toggle('day-mode', isDayMode);
  const css = getComputedStyle(document.body);
  const bg = css.getPropertyValue('--bg-canvas-solid').trim() || '#111827';
  renderer.setClearColor(bg, 1.0);
  applyLightingMode(isDayMode);
  const btn = document.getElementById('btn-theme-mode');
  if (btn) {
    btn.textContent = isDayMode ? '☀' : '☾';
    btn.title = isDayMode ? 'Switch to night mode' : 'Switch to day mode';
    btn.setAttribute('aria-label', btn.title);
  }
  refreshEdgeOverlay();
}

function getLayerKind(layer) {
  return String(layer?.layer_kind || (layer?.points ? 'pointcloud' : 'mesh')).toLowerCase() === 'pointcloud'
    ? 'pointcloud'
    : 'mesh';
}

function getLayerEdgeThickness(layer) {
  const value = Number(layer?.edge_thickness ?? 1.0);
  return Number.isFinite(value) ? Math.min(12.0, Math.max(1.0, value)) : 1.0;
}

function clampOpacity(value, fallback = 1.0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.min(1.0, Math.max(0.0, numeric)) : fallback;
}

function setLineMaterialResolution(material, width = null, height = null) {
  if (!material || !material.resolution) return;
  const viewport = getViewerViewportSize();
  material.resolution.set(width ?? viewport.width, height ?? viewport.height);
}

function refreshLineMaterialResolutions(width = null, height = null) {
  for (const entry of layers) {
    if (entry.edgeLines?.material) {
      setLineMaterialResolution(entry.edgeLines.material, width, height);
    }
    if (entry.pointLines?.material) {
      setLineMaterialResolution(entry.pointLines.material, width, height);
    }
  }
}

function createPositionAttribute(values) {
  const attr = new THREE.BufferAttribute(values, 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  return attr;
}

function setPositionAttribute(geometry, values) {
  const attr = createPositionAttribute(values);
  geometry.setAttribute('position', attr);
  return attr;
}

function setIndexAttribute(geometry, indices) {
  const typed = indices instanceof Uint32Array ? indices : new Uint32Array(indices || []);
  const attr = new THREE.BufferAttribute(typed, 1);
  attr.setUsage(THREE.DynamicDrawUsage);
  geometry.setIndex(attr);
  return attr;
}

function meshFaceSize(layer) {
  return Math.max(3, Math.min(4, Math.floor(Number(layer?.face_size || 3))));
}

function triangulateFaces(faces, faceSize = 3) {
  const width = Math.max(3, Math.min(4, Math.floor(Number(faceSize || 3))));
  if (width === 3) {
    return faces instanceof Uint32Array ? faces : new Uint32Array(faces || []);
  }
  const faceCount = Math.floor((faces?.length || 0) / width);
  const out = new Uint32Array(faceCount * 6);
  for (let faceIdx = 0; faceIdx < faceCount; faceIdx += 1) {
    const src = faceIdx * width;
    const dst = faceIdx * 6;
    const a = faces[src];
    const b = faces[src + 1];
    const c = faces[src + 2];
    const d = faces[src + 3];
    out[dst] = a;
    out[dst + 1] = b;
    out[dst + 2] = c;
    out[dst + 3] = a;
    out[dst + 4] = c;
    out[dst + 5] = d;
  }
  return out;
}

function buildFaceLoopEdgeIndices(
  faces,
  faceSize = 3,
  vertexPassMask = null,
  facePassMask = null,
  faceBegin = 0,
  faceEnd = null,
) {
  const width = Math.max(3, Math.min(4, Math.floor(Number(faceSize || 3))));
  const faceCount = Math.floor((faces?.length || 0) / width);
  const begin = Math.max(0, Math.min(faceCount, Math.floor(faceBegin)));
  const end = Math.max(begin, Math.min(faceCount, Math.floor(faceEnd ?? faceCount)));
  const indices = new Uint32Array((end - begin) * width * 2);
  let writeIdx = 0;
  for (let faceIdx = begin; faceIdx < end; faceIdx += 1) {
    const base = faceIdx * width;
    if (facePassMask && !facePassMask[faceIdx]) continue;
    if (vertexPassMask) {
      let facePasses = true;
      for (let local = 0; local < width; local += 1) {
        facePasses = facePasses && Boolean(vertexPassMask[faces[base + local]]);
      }
      if (!facePasses) continue;
    }
    for (let local = 0; local < width; local += 1) {
      indices[writeIdx] = faces[base + local];
      indices[writeIdx + 1] = faces[base + ((local + 1) % width)];
      writeIdx += 2;
    }
  }
  return writeIdx === indices.length ? indices : indices.slice(0, writeIdx);
}

function expandEdgePositions(vertices, indices) {
  const positions = new Float32Array(indices.length * 3);
  for (let idx = 0; idx < indices.length; idx += 1) {
    const source = indices[idx] * 3;
    const target = idx * 3;
    positions[target] = vertices[source];
    positions[target + 1] = vertices[source + 1];
    positions[target + 2] = vertices[source + 2];
  }
  return positions;
}

function refreshMeshBounds(geometry, computeNormals = true) {
  if (computeNormals) geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}


function createMeshMaterial({
  textureMap = null,
  textureMode = false,
  opacity = 1.0,
  masked = false,
  unlit = false,
} = {}) {
  if (unlit) {
    return new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: masked || opacity < 0.999,
      opacity,
      depthWrite: !masked && opacity > 0,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  }
  if (textureMode && textureMap) {
    return new THREE.MeshLambertMaterial({
      vertexColors: false,
      map: textureMap,
      color: 0xffffff,
      side: THREE.FrontSide,
      transparent: masked || opacity < 0.999,
      opacity,
      depthWrite: !masked && opacity > 0,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  }
  return new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: masked || opacity < 0.999,
    opacity,
    depthWrite: !masked && opacity > 0,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

function buildMesh(layer) {
  const geometry = new THREE.BufferGeometry();
  setPositionAttribute(geometry, layer.vertices);
  geometry.setAttribute('color', createStoredColorAttribute(layer.colors));
  const hasTexture = Boolean(
    layer.texture
    && typeof layer.texture.image_data_uri === 'string'
    && layer.texture.uv
    && layer.texture.uv.length === Math.floor(layer.vertices.length / 3) * 2
  );
  let textureMap = null;
  if (hasTexture) {
    geometry.setAttribute('uv', new THREE.BufferAttribute(layer.texture.uv, 2));
    textureMap = new THREE.TextureLoader().load(layer.texture.image_data_uri);
    textureMap.colorSpace = THREE.SRGBColorSpace;
    textureMap.flipY = true;
    textureMap.wrapS = THREE.ClampToEdgeWrapping;
    textureMap.wrapT = THREE.ClampToEdgeWrapping;
    textureMap.minFilter = THREE.LinearMipmapLinearFilter;
    textureMap.magFilter = THREE.LinearFilter;
    textureMap.anisotropy = renderer.capabilities.getMaxAnisotropy();
    textureMap.needsUpdate = true;
  }
  const textureModeActive = Boolean(textureMap && layer.active_cmap === TEXTURE_CMAP);
  setIndexAttribute(geometry, triangulateFaces(layer.faces, meshFaceSize(layer)));
  refreshMeshBounds(geometry, !layer.unlit);

  const initialOpacity = clampOpacity(layer.opacity ?? payload.opacity);
  const material = createMeshMaterial({
    textureMap,
    textureMode: textureModeActive,
    opacity: initialOpacity,
    unlit: Boolean(layer.unlit),
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;
  scene.add(mesh);

  // Filtering is optional. Keep the masked geometry empty until the user adds
  // a filter instead of eagerly duplicating every vertex/color/UV buffer.
  const maskedGeometry = new THREE.BufferGeometry();
  setPositionAttribute(maskedGeometry, new Float32Array(0));
  maskedGeometry.setAttribute('color', createStoredColorAttribute(new Uint8Array(0)));
  setIndexAttribute(maskedGeometry, new Uint32Array(0));
  const maskedMaterial = createMeshMaterial({
    textureMap,
    textureMode: textureModeActive,
    opacity: 0.0,
    masked: true,
    unlit: Boolean(layer.unlit),
  });
  const maskedMesh = new THREE.Mesh(maskedGeometry, maskedMaterial);
  maskedMesh.frustumCulled = true;
  maskedMesh.visible = false;
  scene.add(maskedMesh);

  return { mesh, maskedMesh, edgeLines: null, textureMap };
}

function shouldUseMeshEdgeDomain(entry) {
  return Boolean(
    entry.layer?.edges?.length
    && (entry.activeFieldAssociation === 'edge' || entry.edgeFilterActive),
  );
}

function buildChunkedMeshTopologyEdges(entry) {
  const layer = entry.layer;
  const faceSize = meshFaceSize(layer);
  const faceCount = Math.floor(layer.faces.length / faceSize);
  const material = new THREE.LineBasicMaterial({
    color: edgeColorHex(),
    transparent: true,
    opacity: isDayMode ? 0.92 : 1.0,
    depthTest: true,
    depthWrite: false,
  });
  const group = new THREE.Group();
  const sharedPositions = new THREE.BufferAttribute(layer.vertices, 3);
  let edgeCount = 0;
  for (let begin = 0; begin < faceCount; begin += MESH_EDGE_CHUNK_FACE_COUNT) {
    const end = Math.min(begin + MESH_EDGE_CHUNK_FACE_COUNT, faceCount);
    const edgeIndices = buildFaceLoopEdgeIndices(
      layer.faces,
      faceSize,
      entry.filterActive ? entry.passMask : null,
      entry.facePassMask || null,
      begin,
      end,
    );
    if (edgeIndices.length === 0) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', sharedPositions);
    geometry.setIndex(new THREE.BufferAttribute(edgeIndices, 1));
    const lines = new THREE.LineSegments(geometry, material);
    lines.frustumCulled = false;
    lines.renderOrder = 3;
    // Keep the source-face interval so a sorted face-field filter can update
    // the mesh and its topology edges with matching GPU draw ranges.
    lines.userData.sourceFaceBegin = begin;
    lines.userData.sourceFaceEnd = end;
    lines.userData.segmentsPerFace = faceSize;
    group.add(lines);
    edgeCount += edgeIndices.length / 2;
  }
  if (edgeCount === 0) {
    material.dispose();
    return null;
  }
  group.material = material;
  group.userData.wideEdges = false;
  group.userData.meshEdgeDomain = false;
  group.userData.contiguousFullTopology = !(
    entry.filterActive && (entry.passMask || entry.facePassMask)
  );
  group.userData.edgeCount = edgeCount;
  group.visible = false;
  group.renderOrder = 3;
  scene.add(group);
  entry.unmaskedEdgeCount = edgeCount;
  return group;
}

function buildChunkedWideMeshTopologyEdges(entry) {
  const layer = entry.layer;
  const faceSize = meshFaceSize(layer);
  const faceCount = Math.floor(layer.faces.length / faceSize);
  const material = new LineMaterial({
    color: edgeColorHex(),
    transparent: true,
    opacity: isDayMode ? 0.92 : 1.0,
    depthTest: true,
    depthWrite: false,
    linewidth: entry.edgeThickness,
  });
  setLineMaterialResolution(material);
  const group = new THREE.Group();
  let edgeCount = 0;
  for (let begin = 0; begin < faceCount; begin += MESH_EDGE_CHUNK_FACE_COUNT) {
    const end = Math.min(begin + MESH_EDGE_CHUNK_FACE_COUNT, faceCount);
    const edgeIndices = buildFaceLoopEdgeIndices(
      layer.faces,
      faceSize,
      entry.filterActive ? entry.passMask : null,
      entry.facePassMask || null,
      begin,
      end,
    );
    if (edgeIndices.length === 0) continue;
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(expandEdgePositions(layer.vertices, edgeIndices));
    geometry.instanceCount = edgeIndices.length / 2;
    const lines = new LineSegments2(geometry, material);
    lines.frustumCulled = false;
    lines.renderOrder = 3;
    group.add(lines);
    edgeCount += edgeIndices.length / 2;
  }
  if (edgeCount === 0) {
    material.dispose();
    return null;
  }
  group.material = material;
  group.userData.wideEdges = true;
  group.userData.meshEdgeDomain = false;
  group.userData.edgeCount = edgeCount;
  group.visible = false;
  group.renderOrder = 3;
  scene.add(group);
  entry.unmaskedEdgeCount = edgeCount;
  return group;
}

function buildMeshEdgeLines(entry) {
  const layer = entry.layer;
  if (shouldUseMeshEdgeDomain(entry)) {
    const edgeColors = entry.activeFieldAssociation === 'edge'
      ? entry.currentEdgeColors
      : null;
    const segmentData = buildPointEdgeSegmentData(
      layer,
      entry.filterActive ? entry.passMask : null,
      entry.edgePassMask || null,
      edgeColors,
    );
    if (segmentData.edgeCount === 0) return null;
    const edgeGeometry = new LineSegmentsGeometry();
    edgeGeometry.setPositions(segmentData.positions);
    if (segmentData.colors) edgeGeometry.setColors(segmentData.colors);
    edgeGeometry.instanceCount = segmentData.edgeCount;
    const edgeMaterial = new LineMaterial({
      color: segmentData.colors ? 0xffffff : edgeColorHex(),
      vertexColors: Boolean(segmentData.colors),
      transparent: true,
      opacity: isDayMode ? 0.92 : 1.0,
      depthTest: true,
      depthWrite: false,
      linewidth: entry.edgeThickness,
    });
    setLineMaterialResolution(edgeMaterial);
    const edgeLines = new LineSegments2(edgeGeometry, edgeMaterial);
    edgeLines.userData.wideEdges = true;
    edgeLines.userData.meshEdgeDomain = true;
    edgeLines.userData.edgeCount = segmentData.edgeCount;
    edgeLines.visible = false;
    edgeLines.frustumCulled = false;
    edgeLines.renderOrder = 3;
    scene.add(edgeLines);
    entry.unmaskedEdgeCount = segmentData.edgeCount;
    return edgeLines;
  }
  const edgeThickness = entry.edgeThickness;
  if (edgeThickness <= 1.0) {
    return buildChunkedMeshTopologyEdges(entry);
  }
  return buildChunkedWideMeshTopologyEdges(entry);
}

function buildMeshVertexPoints(layer, initialPointSize) {
  const pointSize = Math.min(12.0, Math.max(1.0, Number(initialPointSize ?? layer.point_size ?? payload.point_size) || 5.0));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      size: { value: pointSize },
      opacity: { value: 1.0 },
    },
    vertexShader: pointVertexShader,
    fragmentShader: pointFragmentShader,
    transparent: true,
    depthWrite: false,
  });
  const points = new THREE.Points(new THREE.BufferGeometry(), material);
  points.frustumCulled = false;
  points.visible = false;
  scene.add(points);

  const maskedMaterial = material.clone();
  maskedMaterial.uniforms = {
    size: { value: pointSize },
    opacity: { value: 0.0 },
  };
  const maskedPoints = new THREE.Points(new THREE.BufferGeometry(), maskedMaterial);
  maskedPoints.frustumCulled = false;
  maskedPoints.visible = false;
  scene.add(maskedPoints);
  return { points, maskedPoints };
}

function ensureMeshInteractionProxy(entry) {
  if (!entry?.mesh) return null;
  if (!entry.interactionProxy) {
    const sourcePositions = entry.mesh.geometry.getAttribute('position');
    const sourceColors = entry.mesh.geometry.getAttribute('color');
    const stride = Math.max(1, Math.ceil(sourcePositions.count / MAX_MESH_INTERACTION_PROXY_POINTS));
    const proxyCount = Math.ceil(sourcePositions.count / stride);
    const positions = new Float32Array(proxyCount * 3);
    const colors = new Uint8Array(proxyCount * 3);
    let targetIdx = 0;
    for (let sourceIdx = 0; sourceIdx < sourcePositions.count; sourceIdx += stride) {
      positions[targetIdx * 3] = sourcePositions.getX(sourceIdx);
      positions[targetIdx * 3 + 1] = sourcePositions.getY(sourceIdx);
      positions[targetIdx * 3 + 2] = sourcePositions.getZ(sourceIdx);
      colors[targetIdx * 3] = Math.round(sourceColors.getX(sourceIdx) * 255);
      colors[targetIdx * 3 + 1] = Math.round(sourceColors.getY(sourceIdx) * 255);
      colors[targetIdx * 3 + 2] = Math.round(sourceColors.getZ(sourceIdx) * 255);
      targetIdx += 1;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
    geometry.boundingBox = entry.mesh.geometry.boundingBox?.clone() || null;
    geometry.boundingSphere = entry.mesh.geometry.boundingSphere?.clone() || null;
    const material = new THREE.PointsMaterial({
      size: 1.5,
      sizeAttenuation: false,
      vertexColors: true,
    });
    entry.interactionProxy = new THREE.Points(geometry, material);
    entry.interactionProxy.visible = false;
    scene.add(entry.interactionProxy);
  }
  return entry.interactionProxy;
}

function pointGraphEdgeCount(layer) {
  if (Array.isArray(layer.edge_chunks)) {
    return layer.edge_chunks.reduce(
      (total, chunk) => total + Math.floor((chunk?.length || 0) / 2),
      0,
    );
  }
  return Math.floor((layer.edges?.length || 0) / 2);
}

function buildIndexedPointEdgeChunks(layer, positionAttribute) {
  const chunks = Array.isArray(layer.edge_chunks) ? layer.edge_chunks : [];
  if (chunks.length === 0) return null;
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(layer.edge_color || '#000000'),
    opacity: 1.0,
    transparent: false,
  });
  const group = new THREE.Group();
  group.material = material;
  group.userData.indexedPointEdgeChunks = true;
  group.userData.edgeCount = 0;
  for (const indices of chunks) {
    if (!indices?.length) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positionAttribute);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    const lines = new THREE.LineSegments(geometry, material);
    lines.frustumCulled = false;
    group.add(lines);
    group.userData.edgeCount += Math.floor(indices.length / 2);
  }
  group.frustumCulled = false;
  scene.add(group);
  return group;
}

function buildPoints(layer) {
  const geometry = new THREE.BufferGeometry();
  setPositionAttribute(geometry, layer.points);
  geometry.setAttribute('color', createStoredColorAttribute(layer.colors));

  const pointSize = Math.min(12.0, Math.max(1.0, Number(layer.point_size ?? payload.point_size) || 5.0));
  const initialOpacity = clampOpacity(layer.opacity ?? payload.opacity);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      size: { value: pointSize },
      opacity: { value: initialOpacity },
    },
    vertexShader: pointVertexShader,
    fragmentShader: pointFragmentShader,
    transparent: true,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  scene.add(points);

  const maskedGeometry = new THREE.BufferGeometry();
  setPositionAttribute(maskedGeometry, new Float32Array(0));
  maskedGeometry.setAttribute('color', createStoredColorAttribute(new Uint8Array(0)));
  const maskedMaterial = new THREE.ShaderMaterial({
    uniforms: {
      size: { value: pointSize },
      opacity: { value: 0.0 },
    },
    vertexShader: pointVertexShader,
    fragmentShader: pointFragmentShader,
    transparent: true,
    depthWrite: false,
  });
  const maskedPoints = new THREE.Points(maskedGeometry, maskedMaterial);
  maskedPoints.frustumCulled = false;
  maskedPoints.visible = false;
  scene.add(maskedPoints);

  let lines = buildIndexedPointEdgeChunks(layer, geometry.getAttribute('position'));
  if (!lines && layer.edges && layer.edges.length >= 2) {
    const edgeThickness = getLayerEdgeThickness(layer);
    const edgeColors = layer.edge_field_colors?.[layer.active_edge_field_idx || 0] || null;
    const segmentData = buildPointEdgeSegmentData(layer, null, null, edgeColors);
    const { positions, colors, edgeCount } = segmentData;
    if (positions.length > 0) {
      const lineGeometry = new LineSegmentsGeometry();
      lineGeometry.setPositions(positions);
      if (colors) lineGeometry.setColors(colors);
      lineGeometry.instanceCount = edgeCount;
      const lineMaterial = new LineMaterial({
        color: new THREE.Color(colors ? '#ffffff' : (layer.edge_color || '#000000')),
        vertexColors: Boolean(colors),
        opacity: 1.0,
        transparent: false,
        linewidth: edgeThickness,
      });
      setLineMaterialResolution(lineMaterial);
      lines = new LineSegments2(lineGeometry, lineMaterial);
      lines.userData.edgeCount = edgeCount;
      lines.frustumCulled = false;
      scene.add(lines);
    }
  }
  return { points, maskedPoints, lines };
}

function updateMaskedObjectVisibility(entry) {
  const hasMaskedData = (entry.maskedPointCount || 0) > 0 || (entry.maskedFaceCount || 0) > 0;
  const maskedVisible = entry.visible && hasMaskedData && filterState.maskedOpacity > 0.0;
  if (entry.maskedMesh) entry.maskedMesh.visible = maskedVisible;
  if (entry.maskedPoints) entry.maskedPoints.visible = maskedVisible;
  if (entry.maskedVertexPoints) entry.maskedVertexPoints.visible = maskedVisible && entry.showVertices;
}

function updateEntryOpacity(entry, opacity) {
  const nextOpacity = clampOpacity(opacity, entry.opacity ?? 1.0);
  entry.opacity = nextOpacity;
  if (entry.mesh) {
    entry.mesh.material.opacity = nextOpacity;
    entry.mesh.material.transparent = nextOpacity < 0.999;
    // Keep the visible surface occluding hidden wire edges when fading it.
    // Dropping depth writes at 0.999 made the first slider step behave like
    // an X-ray mode. Only a fully hidden surface should stop writing depth.
    entry.mesh.material.depthWrite = nextOpacity > 0;
    entry.mesh.material.needsUpdate = true;
  }
  if (entry.maskedMesh) {
    entry.maskedMesh.material.opacity = nextOpacity * filterState.maskedOpacity;
    entry.maskedMesh.material.transparent = entry.maskedMesh.material.opacity < 0.999;
    entry.maskedMesh.material.needsUpdate = true;
  }
  if (entry.points) {
    entry.points.material.uniforms.opacity.value = nextOpacity;
  }
  if (entry.maskedPoints) {
    entry.maskedPoints.material.uniforms.opacity.value = nextOpacity * filterState.maskedOpacity;
  }
  if (entry.vertexPoints) {
    entry.vertexPoints.material.uniforms.opacity.value = 1.0;
  }
  if (entry.maskedVertexPoints) {
    entry.maskedVertexPoints.material.uniforms.opacity.value = filterState.maskedOpacity;
  }
  if (entry.pointLines) {
    entry.pointLines.material.opacity = 1.0;
    entry.pointLines.material.transparent = false;
  }
  updateMaskedObjectVisibility(entry);
}

function disposeMeshEdgeOverlay(entry) {
  const overlay = entry?.edgeLines;
  if (!overlay) return;
  scene.remove(overlay);
  const materials = new Set();
  overlay.traverse((object3d) => {
    object3d.geometry?.dispose?.();
    if (object3d.material) materials.add(object3d.material);
  });
  if (overlay.material) materials.add(overlay.material);
  materials.forEach((material) => material.dispose?.());
  entry.edgeLines = null;
}

function updateEntryEdgeThickness(entry, thickness) {
  const requestedThickness = Math.min(12.0, Math.max(1.0, Number(thickness) || 1.0));
  const needsWideEdges = requestedThickness > 1.0;
  entry.edgeThickness = requestedThickness;
  if (entry.edgeLines) {
    const hasWideEdges = Boolean(entry.edgeLines.userData.wideEdges);
    if (hasWideEdges !== needsWideEdges) {
      const wasVisible = entry.edgeLines.visible;
      disposeMeshEdgeOverlay(entry);
      entry.edgeLines = buildMeshEdgeLines(entry);
      if (entry.edgeLines) entry.edgeLines.visible = wasVisible;
    } else if (hasWideEdges) {
      entry.edgeLines.material.linewidth = requestedThickness;
    }
  }
  if (entry.pointLines) {
    entry.pointLines.material.linewidth = requestedThickness;
  }
}

function isMeshEdgeFieldActive(entry) {
  return entry.layerKind === 'mesh' && entry.activeFieldAssociation === 'edge';
}

function shouldDisplayMeshEdges(entry) {
  return Boolean(entry.showEdges || isMeshEdgeFieldActive(entry));
}

function updateEntryEdgeOverlay(entry) {
  const displayMeshEdges = shouldDisplayMeshEdges(entry);
  if (entry.layerKind === 'mesh' && displayMeshEdges && !entry.edgeLines) {
    entry.edgeLines = buildMeshEdgeLines(entry);
    updateEntryEdgeThickness(entry, entry.edgeThickness);
  }
  if (entry.sortedFaceFilterActive && entry.edgeLines) {
    applySortedMeshEdgeDrawRange(
      entry,
      entry.sortedFaceFilterBegin,
      entry.sortedFaceFilterEnd,
    );
  }
  if (entry.edgeLines) {
    const color = edgeColorHex();
    const displayedEdgeCount = Number(entry.edgeLines.userData.edgeCount)
      || Number(entry.unmaskedEdgeCount)
      || 0;
    entry.edgeLines.visible = entry.visible && displayMeshEdges && displayedEdgeCount > 0;
    const hasEdgeColors = entry.activeFieldAssociation === 'edge'
      && Boolean(entry.currentEdgeColors?.length);
    entry.edgeLines.material.color.setHex(hasEdgeColors ? 0xffffff : color);
    entry.edgeLines.material.vertexColors = hasEdgeColors;
    entry.edgeLines.material.needsUpdate = true;
    entry.edgeLines.material.opacity = entry.edgeOpacity ?? (isDayMode ? 0.82 : 0.92);
  }
  if (entry.pointLines) {
    entry.pointLines.visible = entry.visible && (entry.unmaskedEdgeCount || 0) > 0;
    const hasPointEdgeColors = entry.activeFieldAssociation === 'edge'
      && Boolean(entry.currentEdgeColors?.length);
    entry.pointLines.material.vertexColors = hasPointEdgeColors;
    entry.pointLines.material.needsUpdate = true;
    if (!hasPointEdgeColors) {
      entry.pointLines.material.color.set(entry.layer.edge_color || '#000000');
    } else {
      entry.pointLines.material.color.set('#ffffff');
    }
    entry.pointLines.material.opacity = 1.0;
    entry.pointLines.material.transparent = false;
  }
  updateEntryVertexOverlay(entry);
}

function rebuildMeshEdgeOverlay(entry) {
  if (!entry || entry.layerKind !== 'mesh') return;
  disposeMeshEdgeOverlay(entry);
  updateEntryEdgeOverlay(entry);
}

function updateEntryVertexOverlay(entry) {
  if (entry.vertexPoints) {
    entry.vertexPoints.visible = entry.visible && entry.showVertices && (entry.unmaskedPointCount || 0) > 0;
  }
  if (entry.maskedVertexPoints) {
    entry.maskedVertexPoints.visible = entry.visible
      && entry.showVertices
      && (entry.maskedPointCount || 0) > 0
      && filterState.maskedOpacity > 0.0;
  }
}

function updateEntryPointSize(entry, pointSize) {
  const value = Math.min(12.0, Math.max(1.0, Number(pointSize) || entry.pointSize || 5.0));
  entry.pointSize = value;
  const pointObjects = [entry.points, entry.maskedPoints, entry.vertexPoints, entry.maskedVertexPoints];
  for (const obj of pointObjects) {
    if (obj) obj.material.uniforms.size.value = value;
  }
}

function refreshEdgeOverlay() {
  for (const entry of layers) {
    updateEntryEdgeOverlay(entry);
  }
}

const UNIFORM_CMAP = 'uniform';
const TEXTURE_CMAP = 'texture';
const POINT_CLOUD_DEFAULT_COLOR = [0, 0, 255];

function getFallbackCmap() {
  return availableCmaps[0] || 'viridis';
}

function isUniformCmap(cmapName) {
  return !cmapName || cmapName === UNIFORM_CMAP;
}

function isTextureCmap(cmapName) {
  return cmapName === TEXTURE_CMAP;
}

function usesScalarCmap(cmapName) {
  return !isUniformCmap(cmapName) && !isTextureCmap(cmapName);
}

function getColormapPreset(cmapName) {
  if (cmapName && colormapPresets[cmapName]) return colormapPresets[cmapName];
  const fallbackName = getFallbackCmap();
  return colormapPresets[fallbackName] || null;
}

function gradientCss(gradient) {
  if (!Array.isArray(gradient) || gradient.length === 0) {
    return 'linear-gradient(90deg, #64748b, #e2e8f0)';
  }
  return `linear-gradient(90deg, ${gradient.join(', ')})`;
}

function getAssociationFieldPrefix(association = 'vertex') {
  if (association === 'edge') return 'edge_field';
  if (association === 'face') return 'face_field';
  return 'field';
}

function getAssociationFieldData(entry, association = 'vertex') {
  const prefix = getAssociationFieldPrefix(association);
  return {
    names: entry.layer[`${prefix}_names`] || [],
    colors: entry.layer[`${prefix}_colors`] || null,
    values: entry.layer[`${prefix}_values`] || null,
    metas: entry.layer[`${prefix}_scalar_meta`] || null,
  };
}

function getAssociationState(entry, association = 'vertex') {
  if (!entry.associationFieldStates) entry.associationFieldStates = {};
  if (!entry.associationFieldStates[association]) {
    const initialIdx = association === 'edge'
      ? (entry.activeEdgeFieldIdx ?? 0)
      : association === 'face'
        ? (entry.activeFaceFieldIdx ?? 0)
        : (entry.activeFieldIdx ?? 0);
    const initialCmap = association === 'vertex'
      ? (entry.activeCmap || UNIFORM_CMAP)
      : (getAssociationFieldData(entry, association).metas?.[initialIdx]?.cmap || getFallbackCmap());
    entry.associationFieldStates[association] = {
      activeFieldIdx: initialIdx,
      activeCmap: initialCmap,
      clipStates: [],
      sharedClip: createClipState(),
      currentColors: null,
    };
  }
  return entry.associationFieldStates[association];
}

function getEntryDisplayedValues(entry, fieldIdx = null, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? state.activeFieldIdx ?? 0;
  const fieldValues = getAssociationFieldData(entry, association).values;
  if (association !== 'vertex') return fieldValues ? fieldValues[resolvedIdx] : null;
  return fieldValues ? fieldValues[resolvedIdx] : entry.layer.values || null;
}

function computeFiniteValueRange(values) {
  if (!values || typeof values.length !== 'number' || values.length === 0) return null;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let idx = 0; idx < values.length; idx += 1) {
    const value = Number(values[idx]);
    if (!Number.isFinite(value)) continue;
    minVal = Math.min(minVal, value);
    maxVal = Math.max(maxVal, value);
  }
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  return { min: minVal, max: maxVal };
}

function sampleRawGradientFromColors(colors, stepCount = 7) {
  const flat = decodeArraySpec(colors, Uint8Array) || new Uint8Array(0);
  const colorCount = Math.floor(flat.length / 3);
  if (colorCount === 0) return null;
  const gradient = [];
  for (let idx = 0; idx < stepCount; idx += 1) {
    const sampleIdx = stepCount <= 1 || colorCount <= 1
      ? 0
      : Math.round((idx * (colorCount - 1)) / (stepCount - 1));
    gradient.push(`rgb(${flat[sampleIdx * 3]}, ${flat[sampleIdx * 3 + 1]}, ${flat[sampleIdx * 3 + 2]})`);
  }
  return gradient;
}

function buildRawScalarGradient(range, stepCount = 7) {
  if (!range) return null;
  const useUnitScale = range.max <= 1.0 + 1e-6;
  const gradient = [];
  for (let idx = 0; idx < stepCount; idx += 1) {
    const t = stepCount <= 1 ? 0.0 : idx / (stepCount - 1);
    const raw = range.min + (range.max - range.min) * t;
    const intensity = useUnitScale ? raw * 255.0 : raw;
    const clipped = Math.max(0, Math.min(255, Math.round(intensity)));
    gradient.push(`rgb(${clipped}, ${clipped}, ${clipped})`);
  }
  return gradient;
}

function getEntryRawColors(entry, fieldIdx = null, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? state.activeFieldIdx ?? 0;
  const fieldColors = getAssociationFieldData(entry, association).colors;
  if (fieldColors && resolvedIdx >= 0 && resolvedIdx < fieldColors.length && fieldColors[resolvedIdx]) {
    return fieldColors[resolvedIdx];
  }
  if (association !== 'vertex') return null;
  return entry.layer.raw_colors || entry.layer.colors || null;
}

function getEntryColorModeOptions(entry, fieldIdx = null, association = 'vertex') {
  const options = [UNIFORM_CMAP];
  if (association === 'vertex' && entry.layer?.texture) {
    options.push(TEXTURE_CMAP);
  }
  const values = getEntryDisplayedValues(entry, fieldIdx, association);
  if (values && typeof values.length === 'number') {
    availableCmaps.forEach((cmapName) => options.push(cmapName));
  }
  return Array.from(new Set(options));
}

function getEntryRawGradient(entry, fieldIdx = null, association = 'vertex') {
  const values = getEntryDisplayedValues(entry, fieldIdx, association);
  const valueRange = computeFiniteValueRange(values);
  if (valueRange) return buildRawScalarGradient(valueRange);
  return sampleRawGradientFromColors(getEntryRawColors(entry, fieldIdx, association));
}

function buildDistributionSvg(values, meta) {
  if (!values || typeof values.length !== 'number' || values.length === 0 || !meta) return null;

  const minVal = Number(meta.min);
  const maxVal = Number(meta.max);
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;

  const bins = 48;
  const counts = new Float32Array(bins);
  const span = maxVal - minVal;
  const hasSpan = Number.isFinite(span) && Math.abs(span) > 1e-12;
  let finiteCount = 0;

  for (let idx = 0; idx < values.length; idx += 1) {
    const raw = Number(values[idx]);
    if (!Number.isFinite(raw)) continue;
    finiteCount += 1;

    let binIdx = 0;
    if (hasSpan) {
      const normalized = Math.min(1, Math.max(0, (raw - minVal) / span));
      binIdx = Math.min(bins - 1, Math.floor(normalized * bins));
    } else {
      binIdx = Math.floor((bins - 1) * 0.5);
    }
    counts[binIdx] += 1;
  }

  if (finiteCount === 0) return null;

  let peak = 0;
  for (let idx = 0; idx < bins; idx += 1) {
    peak = Math.max(peak, counts[idx]);
  }
  if (!(peak > 0)) return null;

  const width = 100;
  const height = 32;
  const baselineY = 30;
  const curveTopY = 3;
  const curveHeight = baselineY - curveTopY;
  const strokeInset = 0.7;
  const xStep = (width - 2 * strokeInset) / (bins - 1);

  let linePath = '';
  for (let idx = 0; idx < bins; idx += 1) {
    const x = strokeInset + idx * xStep;
    const y = baselineY - (counts[idx] / peak) * curveHeight;
    linePath += `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)} `;
  }

  const areaPath = `${linePath}L ${(width - strokeInset).toFixed(2)} ${baselineY} L ${strokeInset.toFixed(2)} ${baselineY} Z`;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.classList.add('colorbar-distribution');

  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const clipIdBase = `dist-clip-${Math.random().toString(36).slice(2)}`;

  const centerClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
  centerClip.setAttribute('id', `${clipIdBase}-center`);
  const centerRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  centerRect.setAttribute('y', '0');
  centerRect.setAttribute('height', String(height));
  centerClip.appendChild(centerRect);

  const leftClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
  leftClip.setAttribute('id', `${clipIdBase}-left`);
  const leftRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  leftRect.setAttribute('x', '0');
  leftRect.setAttribute('y', '0');
  leftRect.setAttribute('height', String(height));
  leftClip.appendChild(leftRect);

  const rightClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
  rightClip.setAttribute('id', `${clipIdBase}-right`);
  const rightRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rightRect.setAttribute('y', '0');
  rightRect.setAttribute('height', String(height));
  rightClip.appendChild(rightRect);

  defs.appendChild(centerClip);
  defs.appendChild(leftClip);
  defs.appendChild(rightClip);
  svg.appendChild(defs);

  const buildGroup = (fillClass, lineClass, clipId) => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('clip-path', `url(#${clipId})`);

    const fillPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    fillPath.setAttribute('d', areaPath.trim());
    fillPath.setAttribute('class', fillClass);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.setAttribute('d', linePath.trim());
    line.setAttribute('class', lineClass);

    group.appendChild(fillPath);
    group.appendChild(line);
    return group;
  };

  svg.appendChild(buildGroup('colorbar-distribution-fill faded', 'colorbar-distribution-line faded', `${clipIdBase}-left`));
  svg.appendChild(buildGroup('colorbar-distribution-fill', 'colorbar-distribution-line', `${clipIdBase}-center`));
  svg.appendChild(buildGroup('colorbar-distribution-fill faded', 'colorbar-distribution-line faded', `${clipIdBase}-right`));

  const setClipFractions = (startFrac = 0.0, endFrac = 1.0) => {
    const start = Math.min(1.0, Math.max(0.0, Number(startFrac) || 0.0));
    const end = Math.min(1.0, Math.max(start, Number(endFrac) || 0.0));
    const leftWidth = width * start;
    const centerX = leftWidth;
    const centerWidth = width * Math.max(0.0, end - start);
    const rightX = width * end;
    const rightWidth = Math.max(0.0, width - rightX);

    leftRect.setAttribute('width', leftWidth.toFixed(2));
    centerRect.setAttribute('x', centerX.toFixed(2));
    centerRect.setAttribute('width', centerWidth.toFixed(2));
    rightRect.setAttribute('x', rightX.toFixed(2));
    rightRect.setAttribute('width', rightWidth.toFixed(2));
  };

  setClipFractions(0.0, 1.0);
  return { svg, setClipFractions };
}

function createCmapSwatch(entry, fieldIdx, cmapName, association = 'vertex') {
  const swatch = document.createElement('span');
  swatch.className = 'cmap-swatch';
  if (isTextureCmap(cmapName)) {
    swatch.style.background = 'linear-gradient(135deg, #f8fafc 0%, #f8fafc 34%, #0f172a 35%, #0f172a 43%, #dc2626 44%, #dc2626 56%, #2563eb 57%, #2563eb 66%, #f8fafc 67%, #f8fafc 100%)';
    return swatch;
  }
  const gradient = isUniformCmap(cmapName)
    ? getEntryRawGradient(entry, fieldIdx, association)
    : (getColormapPreset(cmapName)?.gradient || null);
  swatch.style.background = gradientCss(gradient);
  return swatch;
}

function setCmapButtonContent(button, cmapName) {
  button.replaceChildren();
  const prefix = document.createElement('span');
  prefix.className = 'dropdown-label';
  prefix.textContent = 'Colormap:';
  const value = document.createElement('span');
  value.className = 'layer-id';
  value.textContent = cmapName || UNIFORM_CMAP;
  button.appendChild(prefix);
  button.appendChild(value);
}

function setFieldButtonContent(button, fieldName) {
  button.replaceChildren();
  const prefix = document.createElement('span');
  prefix.className = 'dropdown-label field-dropdown-label';
  prefix.textContent = 'Field:';
  const value = document.createElement('span');
  value.className = 'layer-id';
  value.textContent = fieldName;
  button.appendChild(prefix);
  button.appendChild(value);
}

function setFilterDropdownButtonContent(button, prefixText, valueText) {
  button.replaceChildren();
  const prefix = document.createElement('span');
  prefix.className = 'dropdown-label field-dropdown-label';
  prefix.textContent = prefixText;
  const value = document.createElement('span');
  value.className = 'layer-id';
  value.textContent = valueText;
  button.appendChild(prefix);
  button.appendChild(value);
}

function createFilterDropdownControl({
  prefixText,
  options,
  selectedValue,
  onSelect,
  hostClassName = '',
}) {
  const wrap = document.createElement('div');
  wrap.className = `layer-select-wrapper ${hostClassName}`.trim();
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-dropdown';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'layer-select-button';
  const menu = document.createElement('div');
  menu.className = 'custom-dropdown-menu';

  const selectedOption = options.find((option) => option.value === selectedValue) || options[0];
  setFilterDropdownButtonContent(button, prefixText, selectedOption ? selectedOption.label : '');

  options.forEach((option) => {
    const item = document.createElement('div');
    item.className = 'custom-dropdown-item';
    if (option.value === selectedValue) item.classList.add('selected');
    const value = document.createElement('span');
    value.className = 'layer-id';
    value.textContent = option.label;
    item.appendChild(value);
    item.addEventListener('click', () => {
      setFilterDropdownButtonContent(button, prefixText, option.label);
      menu.querySelectorAll('.custom-dropdown-item').forEach((el) => {
        el.classList.toggle('selected', el === item);
      });
      onSelect(option.value);
      closeDropdownMenu(menu);
    });
    menu.appendChild(item);
  });

  button.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleDropdownMenu(button, menu);
  });

  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  wrap.appendChild(dropdown);
  return wrap;
}

function summarizeMultiSelectOptions(options, selectedSet) {
  if (selectedSet.has(FILTER_SCOPE_ALL_KEY)) return 'All';
  const selected = options.filter((option) => selectedSet.has(option.value));
  if (selected.length === 0) return 'None';
  if (selected.length === options.length) return 'All';
  if (selected.length === 1) return selected[0].label;
  return `${selected.length} selected`;
}

function createFilterMultiSelectDropdownControl({
  prefixText,
  options,
  selectedValues,
  onChange,
  hostClassName = '',
}) {
  const wrap = document.createElement('div');
  wrap.className = `layer-select-wrapper ${hostClassName}`.trim();
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-dropdown';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'layer-select-button';
  const menu = document.createElement('div');
  menu.className = 'custom-dropdown-menu';

  const validValues = new Set(options.map((option) => option.value));
  validValues.add(FILTER_SCOPE_ALL_KEY);
  const fallbackToAll = !Array.isArray(selectedValues);
  const selectedSet = new Set(
    normalizeRuleScopeSelection(selectedValues, fallbackToAll).filter((value) => validValues.has(value)),
  );
  const items = [];

  const refreshButtonLabel = () => {
    setFilterDropdownButtonContent(
      button,
      prefixText,
      summarizeMultiSelectOptions(options, selectedSet),
    );
  };
  const refreshItems = () => {
    const allActive = selectedSet.has(FILTER_SCOPE_ALL_KEY);
    items.forEach(({ value, checkbox, item }) => {
      const checked = allActive ? (value === FILTER_SCOPE_ALL_KEY) : selectedSet.has(value);
      checkbox.checked = checked;
      item.classList.toggle('selected', checked);
    });
  };
  const applySelection = () => {
    refreshButtonLabel();
    refreshItems();
    onChange(Array.from(selectedSet));
  };

  const menuOptions = [{ value: FILTER_SCOPE_ALL_KEY, label: 'All' }, ...options];
  menuOptions.forEach((option) => {
    const optionValue = option.value;
    const item = document.createElement('div');
    item.className = 'custom-dropdown-item filter-multiselect-item';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'filter-multiselect-checkbox';

    const value = document.createElement('span');
    value.className = 'layer-id';
    value.textContent = option.label;
    item.appendChild(checkbox);
    item.appendChild(value);

    const applyToggle = (nextChecked) => {
      if (optionValue === FILTER_SCOPE_ALL_KEY) {
        if (nextChecked) {
          selectedSet.clear();
          selectedSet.add(FILTER_SCOPE_ALL_KEY);
        } else {
          selectedSet.delete(FILTER_SCOPE_ALL_KEY);
        }
      } else if (nextChecked) {
        selectedSet.delete(FILTER_SCOPE_ALL_KEY);
        selectedSet.add(optionValue);
      } else {
        selectedSet.delete(optionValue);
      }
      applySelection();
    };

    checkbox.addEventListener('click', (ev) => {
      ev.stopPropagation();
    });
    checkbox.addEventListener('change', (ev) => {
      ev.stopPropagation();
      applyToggle(checkbox.checked);
    });
    item.addEventListener('click', (ev) => {
      ev.stopPropagation();
      applyToggle(!checkbox.checked);
    });
    items.push({ value: optionValue, checkbox, item });
    menu.appendChild(item);
  });

  refreshButtonLabel();
  refreshItems();

  button.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleDropdownMenu(button, menu);
  });

  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  wrap.appendChild(dropdown);
  return wrap;
}

function closeDropdownMenu(menu) {
  if (dropdownPortal && menu.parentElement === dropdownPortal) {
    const host = menu.__hostElement || null;
    if (host) {
      host.appendChild(menu);
    }
  }
  menu.classList.remove('floating');
  menu.classList.remove('open');
  menu.style.top = '';
  menu.style.left = '';
  menu.style.width = '';
  menu.style.maxHeight = '';
}

function closeAllDropdownMenus(exceptMenu = null) {
  document.querySelectorAll('.custom-dropdown-menu.open').forEach((node) => {
    if (node !== exceptMenu) closeDropdownMenu(node);
  });
}

function positionDropdownMenu(button, menu) {
  const rect = button.getBoundingClientRect();
  const margin = 12;
  const gap = 4;
  const menuWidth = rect.width;
  const left = Math.min(
    window.innerWidth - menuWidth - margin,
    Math.max(margin, rect.left),
  );
  const spaceBelow = window.innerHeight - rect.bottom - margin - gap;
  const spaceAbove = rect.top - margin - gap;
  const openUpward = spaceBelow < 220 && spaceAbove > spaceBelow;
  const availableHeight = openUpward ? spaceAbove : spaceBelow;
  const maxHeight = Math.max(140, Math.min(420, availableHeight));

  menu.style.left = `${left}px`;
  menu.style.width = `${menuWidth}px`;
  menu.style.maxHeight = `${maxHeight}px`;

  if (openUpward) {
    const menuHeight = Math.min(menu.scrollHeight || maxHeight, maxHeight);
    menu.style.top = `${Math.max(margin, rect.top - menuHeight - gap)}px`;
  } else {
    menu.style.top = `${Math.min(
      window.innerHeight - maxHeight - margin,
      Math.max(margin, rect.bottom + gap),
    )}px`;
  }
}

function toggleDropdownMenu(button, menu) {
  const willOpen = !menu.classList.contains('open');
  closeAllDropdownMenus(willOpen ? menu : null);
  if (!willOpen) {
    closeDropdownMenu(menu);
    return;
  }
  if (dropdownPortal && menu.parentElement !== dropdownPortal) {
    menu.__hostElement = menu.parentElement;
    dropdownPortal.appendChild(menu);
  }
  menu.classList.add('floating');
  menu.classList.add('open');
  positionDropdownMenu(button, menu);
}

function getEntryCmap(entry, fieldIdx = null, association = 'vertex') {
  return getAssociationState(entry, association).activeCmap || UNIFORM_CMAP;
}

function getEntryScaleRange(entry, fieldIdx = null, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? state.activeFieldIdx ?? 0;
  const values = getEntryDisplayedValues(entry, resolvedIdx, association);
  const fieldMetas = getAssociationFieldData(entry, association).metas;
  const baseMeta = fieldMetas ? fieldMetas[resolvedIdx] : (association === 'vertex' ? entry.layer.scalar_meta : null);
  const metaMin = Number(baseMeta && baseMeta.min);
  const metaMax = Number(baseMeta && baseMeta.max);
  if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {
    return { min: metaMin, max: metaMax };
  }
  return computeFiniteValueRange(values);
}

function getEntryAllScaleRanges(entry, association = 'vertex') {
  const ranges = [];
  const data = getAssociationFieldData(entry, association);
  const fieldValues = data.values;
  const fieldMetas = data.metas;
  if (fieldValues && fieldValues.length > 0) {
    for (let fieldIdx = 0; fieldIdx < fieldValues.length; fieldIdx += 1) {
      const values = fieldValues[fieldIdx];
      const meta = fieldMetas ? fieldMetas[fieldIdx] : null;
      const metaMin = Number(meta && meta.min);
      const metaMax = Number(meta && meta.max);
      if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {
        ranges.push({ min: metaMin, max: metaMax });
        continue;
      }
      const range = computeFiniteValueRange(values);
      if (range) ranges.push(range);
    }
    return ranges;
  }
  const range = getEntryScaleRange(entry, null, association);
  return range ? [range] : [];
}

function getSharedFieldScaleRange(entry, association = 'vertex') {
  if (!entry || !usesScalarCmap(getEntryCmap(entry, null, association))) return null;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (const range of getEntryAllScaleRanges(entry, association)) {
    minVal = Math.min(minVal, range.min);
    maxVal = Math.max(maxVal, range.max);
  }
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  return { min: minVal, max: maxVal };
}

function getScalarMeta(entry, fieldIdx = null, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? state.activeFieldIdx ?? 0;
  const values = getEntryDisplayedValues(entry, resolvedIdx, association);
  const fieldMetas = getAssociationFieldData(entry, association).metas;
  const baseMeta = fieldMetas ? fieldMetas[resolvedIdx] : (association === 'vertex' ? entry.layer.scalar_meta : null);
  const cmapName = getEntryCmap(entry, resolvedIdx, association);
  if (!baseMeta) {
    const range = computeFiniteValueRange(values);
    if (!range) return null;
    const resolved = {
      min: range.min,
      max: range.max,
      cmap: cmapName,
      gradient: usesScalarCmap(cmapName)
        ? (getColormapPreset(cmapName)?.gradient || null)
        : getEntryRawGradient(entry, resolvedIdx, association),
    };
    const sharedFieldScaleRange = getSharedFieldScaleRange(entry, association);
    if (usesScalarCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {
      resolved.min = sharedFieldScaleRange.min;
      resolved.max = sharedFieldScaleRange.max;
    }
    resolved.fullMin = resolved.min;
    resolved.fullMax = resolved.max;
    if (usesScalarCmap(cmapName)) {
      const clippedRange = applyClipToRange(
        { min: resolved.fullMin, max: resolved.fullMax },
        getClipState(entry, resolvedIdx, association),
      );
      if (clippedRange) {
        resolved.min = clippedRange.min;
        resolved.max = clippedRange.max;
      }
    }
    resolved.clipState = createClipState(
      getClipState(entry, resolvedIdx, association).minFrac,
      getClipState(entry, resolvedIdx, association).maxFrac,
    );
    resolved.clipEnabled = usesScalarCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
    return resolved;
  }
  const resolved = {
    ...baseMeta,
    cmap: cmapName,
    gradient: usesScalarCmap(cmapName)
      ? (getColormapPreset(cmapName)?.gradient || baseMeta.gradient)
      : (baseMeta.gradient || getEntryRawGradient(entry, resolvedIdx, association)),
  };
  const sharedFieldScaleRange = getSharedFieldScaleRange(entry, association);
  if (usesScalarCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {
    resolved.min = sharedFieldScaleRange.min;
    resolved.max = sharedFieldScaleRange.max;
  }
  resolved.fullMin = resolved.min;
  resolved.fullMax = resolved.max;
  if (usesScalarCmap(cmapName)) {
    const clippedRange = applyClipToRange(
      { min: resolved.fullMin, max: resolved.fullMax },
      getClipState(entry, resolvedIdx, association),
    );
    if (clippedRange) {
      resolved.min = clippedRange.min;
      resolved.max = clippedRange.max;
    }
  }
  resolved.clipState = createClipState(
    getClipState(entry, resolvedIdx, association).minFrac,
    getClipState(entry, resolvedIdx, association).maxFrac,
  );
  resolved.clipEnabled = usesScalarCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
  return resolved;
}

function buildColorsFromValues(values, meta) {
  const preset = getColormapPreset(meta && meta.cmap);
  const samples = preset && Array.isArray(preset.samples) ? preset.samples : null;
  if (!samples || samples.length === 0 || !values || typeof values.length !== 'number') return null;

  const colorCount = values.length * 3;
  const colors = new Uint8Array(colorCount);
  const minVal = Number.isFinite(meta.min) ? meta.min : 0.0;
  const maxVal = Number.isFinite(meta.max) ? meta.max : 1.0;
  const denom = Math.max(maxVal - minVal, 1e-8);
  const sampleMaxIdx = samples.length - 1;

  for (let idx = 0; idx < values.length; idx += 1) {
    const raw = Number(values[idx]);
    const normalized = Number.isFinite(raw) ? Math.min(1, Math.max(0, (raw - minVal) / denom)) : 0.0;
    const scaled = normalized * sampleMaxIdx;
    const leftIdx = Math.floor(scaled);
    const rightIdx = Math.min(sampleMaxIdx, leftIdx + 1);
    const mix = scaled - leftIdx;
    const left = samples[leftIdx];
    const right = samples[rightIdx];
    colors[(idx * 3)] = Math.round(left[0] + (right[0] - left[0]) * mix);
    colors[(idx * 3) + 1] = Math.round(left[1] + (right[1] - left[1]) * mix);
    colors[(idx * 3) + 2] = Math.round(left[2] + (right[2] - left[2]) * mix);
  }
  return colors;
}

function resolveEntryRawColors(entry, fieldIdx = null, association = 'vertex') {
  const rawColors = getEntryRawColors(entry, fieldIdx, association);
  const decoded = decodeArraySpec(rawColors, Uint8Array);
  if (decoded && decoded.length > 0) return decoded;
  return association === 'vertex'
    ? (decodeArraySpec(entry.layer.colors, Uint8Array) || new Uint8Array(0))
    : new Uint8Array(0);
}

function setMeshTextureMode(entry, enabled) {
  if (!entry || !entry.textureMap) return;
  const replaceMaterial = (object3d, masked = false) => {
    if (!object3d?.material) return;
    const oldMaterial = object3d.material;
    const nextMaterial = createMeshMaterial({
      textureMap: entry.textureMap,
      textureMode: enabled,
      opacity: oldMaterial.opacity,
      masked,
      unlit: Boolean(entry.layer.unlit) && !enabled,
    });
    object3d.material = nextMaterial;
    oldMaterial.dispose?.();
  };
  replaceMaterial(entry.mesh, false);
  replaceMaterial(entry.maskedMesh, true);
}

function setMeshFaceColorMode(entry, enabled) {
  const nextEnabled = Boolean(enabled);
  if (Boolean(entry.faceColorMaterialActive) === nextEnabled) return;
  const replaceMaterial = (object3d, masked = false) => {
    if (!object3d?.material) return;
    const oldMaterial = object3d.material;
    object3d.material = createMeshMaterial({
      opacity: oldMaterial.opacity,
      masked,
      unlit: nextEnabled || Boolean(entry.layer.unlit),
    });
    oldMaterial.dispose?.();
  };
  replaceMaterial(entry.mesh, false);
  replaceMaterial(entry.maskedMesh, true);
  entry.faceColorMaterialActive = nextEnabled;
}

function getEntryDefaultColors(entry) {
  return entry.layer.default_colors
    || entry.layer.raw_colors
    || entry.layer.colors
    || new Uint8Array(0);
}

function syncActiveFieldRadios() {
  document.querySelectorAll('input[data-field-association]').forEach((input) => {
    const entry = layers.find((candidate) => String(candidate.layerId) === input.dataset.layerId);
    input.checked = Boolean(entry && entry.activeFieldAssociation === input.dataset.association);
  });
}

function clearEntryFieldDisplay(entry, requestUpdate = true) {
  entry.activeFieldAssociation = null;
  entry.activeSurfaceAssociation = null;
  entry.activeValues = null;
  entry.currentColors = getEntryDefaultColors(entry);
  entry.currentEdgeColors = null;
  entry.currentFaceColors = null;
  if (entry.pointLines?.material) {
    entry.pointLines.material.vertexColors = false;
    entry.pointLines.material.needsUpdate = true;
  }
  if (entry.layerKind === 'mesh') {
    setMeshFaceColorMode(entry, false);
    setMeshTextureMode(entry, false);
  }
  applyFiltersToEntry(entry);
  syncActiveFieldRadios();
  if (requestUpdate) requestRender();
}

function renderEntryColors(entry, fieldIdx = null, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const resolvedIdx = fieldIdx ?? state.activeFieldIdx ?? 0;
  const values = getEntryDisplayedValues(entry, resolvedIdx, association);
  const rawColors = resolveEntryRawColors(entry, resolvedIdx, association);
  const cmapName = getEntryCmap(entry, resolvedIdx, association);
  const meta = getScalarMeta(entry, resolvedIdx, association);
  const colorFromValues = usesScalarCmap(cmapName) && values && meta ? buildColorsFromValues(values, meta) : null;

  state.activeFieldIdx = resolvedIdx;
  state.activeCmap = cmapName;
  state.currentColors = colorFromValues || rawColors;
  entry.activeFieldAssociation = association;
  if (association !== 'edge') {
    entry.currentEdgeColors = null;
    if (entry.pointLines?.material) {
      entry.pointLines.material.vertexColors = false;
      entry.pointLines.material.needsUpdate = true;
    }
  }
  if (association !== 'face') entry.currentFaceColors = null;
  if (association === 'edge') {
    entry.activeEdgeFieldIdx = resolvedIdx;
    entry.currentEdgeColors = state.currentColors;
    entry.activeSurfaceAssociation = null;
    entry.activeValues = null;
    entry.currentColors = getEntryDefaultColors(entry);
    if (entry.layerKind === 'mesh') {
      setMeshFaceColorMode(entry, false);
      setMeshTextureMode(entry, false);
    }
    if (entry.pointLines) {
      entry.pointLines.material.vertexColors = Boolean(entry.currentEdgeColors?.length);
      entry.pointLines.material.needsUpdate = true;
    }
  } else if (association === 'face') {
    entry.activeFaceFieldIdx = resolvedIdx;
    entry.currentFaceColors = state.currentColors;
    entry.activeSurfaceAssociation = 'face';
  } else {
    entry.activeFieldIdx = resolvedIdx;
    entry.activeValues = values;
    entry.activeCmap = cmapName;
    entry.currentColors = state.currentColors;
    entry.activeSurfaceAssociation = 'vertex';
    setMeshFaceColorMode(entry, false);
    setMeshTextureMode(entry, isTextureCmap(cmapName));
  }
  applyFiltersToEntry(entry);
  syncActiveFieldRadios();
  requestRender();
}

function setLayerCmap(entry, cmapName, association = 'vertex') {
  const previousSurfaceAssociation = entry.activeSurfaceAssociation;
  const state = getAssociationState(entry, association);
  const fieldIdx = state.activeFieldIdx ?? 0;
  state.activeCmap = cmapName;
  if (association === 'vertex') entry.activeCmap = cmapName;
  renderEntryColors(entry, fieldIdx, association);
  updateEntryOpacity(entry, entry.opacity);
  if (
    entry.layerKind === 'mesh'
    && association !== 'edge'
    && previousSurfaceAssociation !== entry.activeSurfaceAssociation
  ) {
    rebuildLayerList();
  } else {
    updateColorbars();
  }
}

function setLayerShareFieldScale(entry, enabled) {
  entry.shareFieldScale = Boolean(enabled);
  const association = entry.activeFieldAssociation;
  if (association && getEntryFieldCount(entry, association) > 0) {
    const state = getAssociationState(entry, association);
    const fieldIdx = Math.max(0, Math.min(state.activeFieldIdx ?? 0, getEntryFieldCount(entry, association) - 1));
    renderEntryColors(entry, fieldIdx, association);
  }
  updateColorbars();
}

function setColorClipBoundary(entry, fieldIdx, boundary, value, association = 'vertex') {
  const previousSurfaceAssociation = entry.activeSurfaceAssociation;
  const state = getClipState(entry, fieldIdx, association);
  setClipBoundary(state, boundary, sliderValueToClip(value, boundary === 'max' ? 1.0 : 0.0));
  renderEntryColors(entry, fieldIdx, association);
  if (
    entry.layerKind === 'mesh'
    && association !== 'edge'
    && previousSurfaceAssociation !== entry.activeSurfaceAssociation
  ) {
    rebuildLayerList();
  }
  syncAllColorbarClipControls();
}

function getEntryScopeKey(entry) {
  return `layer:${entry.layerId}`;
}

function getAllScopeKeys() {
  return layers.map((entry) => getEntryScopeKey(entry));
}

function normalizeRuleScopeSelection(scopeKeys, fallbackToAll = true) {
  const validKeys = new Set(getAllScopeKeys());
  if (!Array.isArray(scopeKeys)) {
    return fallbackToAll ? [FILTER_SCOPE_ALL_KEY] : [];
  }
  if (scopeKeys.includes(FILTER_SCOPE_ALL_KEY)) {
    return [FILTER_SCOPE_ALL_KEY];
  }
  const resolved = [];
  scopeKeys.forEach((key) => {
    if (!validKeys.has(key)) return;
    if (resolved.includes(key)) return;
    resolved.push(key);
  });
  if (resolved.length === 0 && fallbackToAll) {
    return [FILTER_SCOPE_ALL_KEY];
  }
  return resolved;
}

function resolveScopeKeys(scopeKeys, fallbackToAll = true) {
  const normalized = normalizeRuleScopeSelection(scopeKeys, fallbackToAll);
  const validKeys = new Set(getAllScopeKeys());
  if (normalized.includes(FILTER_SCOPE_ALL_KEY)) {
    return Array.from(validKeys);
  }
  return normalized.filter((key) => validKeys.has(key));
}

function getScopedEntries(scopeKeys, fallbackToAll = true) {
  const resolved = resolveScopeKeys(scopeKeys, fallbackToAll);
  if (resolved.length === 0) return [];
  const keySet = new Set(resolved);
  return layers.filter((entry) => keySet.has(getEntryScopeKey(entry)));
}

function isEntryWithinScope(entry, scopeKeys, fallbackToAll = true) {
  const resolved = resolveScopeKeys(scopeKeys, fallbackToAll);
  if (resolved.length === 0) return false;
  return resolved.includes(getEntryScopeKey(entry));
}

function getEntryPositions(entry) {
  return entry.layerKind === 'pointcloud' ? entry.layer.points : entry.layer.vertices;
}

function getEntryElementCount(entry) {
  const positions = getEntryPositions(entry);
  return Math.floor((positions && positions.length ? positions.length : 0) / 3);
}

function getEntryFieldValuesByName(entry, fieldName) {
  if (!fieldName || !entry.fieldLookup) return null;
  return entry.fieldLookup.get(fieldName) || null;
}

function getEntryEdgeFieldValuesByName(entry, fieldName) {
  if (!fieldName || !entry.edgeFieldLookup) return null;
  return entry.edgeFieldLookup.get(fieldName) || null;
}

function getEntryFaceFieldValuesByName(entry, fieldName) {
  if (!fieldName || !entry.faceFieldLookup) return null;
  return entry.faceFieldLookup.get(fieldName) || null;
}

function getEntryFieldRangeByName(entry, fieldName, association = 'vertex') {
  const data = getAssociationFieldData(entry, association);
  const fieldIdx = data.names.findIndex((name) => String(name) === String(fieldName));
  if (fieldIdx < 0) return null;
  const meta = data.metas?.[fieldIdx] || null;
  const metaMin = Number(meta?.min);
  const metaMax = Number(meta?.max);
  if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {
    return { min: metaMin, max: metaMax };
  }
  return computeFiniteValueRange(data.values?.[fieldIdx]);
}

function computeFiniteAxisRange(positions, axisIdx) {
  if (!positions || positions.length === 0) return null;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let idx = axisIdx; idx < positions.length; idx += 3) {
    const value = Number(positions[idx]);
    if (!Number.isFinite(value)) continue;
    minVal = Math.min(minVal, value);
    maxVal = Math.max(maxVal, value);
  }
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  return { min: minVal, max: maxVal };
}

function computeScopePositionRanges(scopeKeys) {
  const ranges = {
    x: { min: Infinity, max: -Infinity },
    y: { min: Infinity, max: -Infinity },
    z: { min: Infinity, max: -Infinity },
  };
  let hasValue = false;
  for (const entry of getScopedEntries(scopeKeys, false)) {
    const positions = getEntryPositions(entry);
    const x = computeFiniteAxisRange(positions, 0);
    const y = computeFiniteAxisRange(positions, 1);
    const z = computeFiniteAxisRange(positions, 2);
    if (!x || !y || !z) continue;
    hasValue = true;
    ranges.x.min = Math.min(ranges.x.min, x.min);
    ranges.x.max = Math.max(ranges.x.max, x.max);
    ranges.y.min = Math.min(ranges.y.min, y.min);
    ranges.y.max = Math.max(ranges.y.max, y.max);
    ranges.z.min = Math.min(ranges.z.min, z.min);
    ranges.z.max = Math.max(ranges.z.max, z.max);
  }
  if (!hasValue) return null;
  return ranges;
}

function computeScopeFieldRange(scopeKey, fieldName) {
  let minVal = Infinity;
  let maxVal = -Infinity;
  let hasValue = false;
  for (const entry of getScopedEntries(scopeKey, false)) {
    const range = getEntryFieldRangeByName(entry, fieldName, 'vertex');
    if (!range) continue;
    hasValue = true;
    minVal = Math.min(minVal, range.min);
    maxVal = Math.max(maxVal, range.max);
  }
  if (!hasValue) return null;
  return { min: minVal, max: maxVal };
}

function computeScopeEdgeFieldRange(scopeKey, fieldName) {
  let minVal = Infinity;
  let maxVal = -Infinity;
  let hasValue = false;
  for (const entry of getScopedEntries(scopeKey, false)) {
    const range = getEntryFieldRangeByName(entry, fieldName, 'edge');
    if (!range) continue;
    hasValue = true;
    minVal = Math.min(minVal, range.min);
    maxVal = Math.max(maxVal, range.max);
  }
  return hasValue ? { min: minVal, max: maxVal } : null;
}

function computeScopeFaceFieldRange(scopeKey, fieldName) {
  let minVal = Infinity;
  let maxVal = -Infinity;
  let hasValue = false;
  for (const entry of getScopedEntries(scopeKey, false)) {
    const range = getEntryFieldRangeByName(entry, fieldName, 'face');
    if (!range) continue;
    hasValue = true;
    minVal = Math.min(minVal, range.min);
    maxVal = Math.max(maxVal, range.max);
  }
  return hasValue ? { min: minVal, max: maxVal } : null;
}

function getFilterFieldNames() {
  const names = new Set();
  for (const entry of layers) {
    const fieldNames = entry.layer.field_names || [];
    fieldNames.forEach((name) => names.add(String(name)));
  }
  return Array.from(names);
}

function getFilterEdgeFieldNames() {
  const names = new Set();
  for (const entry of layers) {
    const fieldNames = entry.layer.edge_field_names || [];
    fieldNames.forEach((name) => names.add(String(name)));
  }
  return Array.from(names);
}

function getFilterFaceFieldNames() {
  const names = new Set();
  for (const entry of layers) {
    const fieldNames = entry.layer.face_field_names || [];
    fieldNames.forEach((name) => names.add(String(name)));
  }
  return Array.from(names);
}

function getFilterPropertyOptions() {
  const options = [{ value: FILTER_POSITION, label: 'Position (x/y/z)' }];
  getFilterFieldNames().forEach((fieldName) => {
    options.push({ value: `field:${fieldName}`, label: `Point/Vertex: ${fieldName}` });
  });
  getFilterEdgeFieldNames().forEach((fieldName) => {
    options.push({ value: `edge-field:${fieldName}`, label: `Edge: ${fieldName}` });
  });
  getFilterFaceFieldNames().forEach((fieldName) => {
    options.push({ value: `face-field:${fieldName}`, label: `Triangle/Face: ${fieldName}` });
  });
  return options;
}

function getRuleDomain(rule, scopeKeys) {
  if (!rule || !rule.property) return null;
  if (rule.property === FILTER_POSITION) {
    const ranges = computeScopePositionRanges(scopeKeys);
    if (!ranges) return null;
    return { kind: FILTER_POSITION, ranges };
  }
  if (rule.property.startsWith('field:')) {
    const fieldName = rule.property.slice(6);
    const range = computeScopeFieldRange(scopeKeys, fieldName);
    if (!range) return null;
    return { kind: 'field', fieldName, range };
  }
  if (rule.property.startsWith('edge-field:')) {
    const fieldName = rule.property.slice(11);
    const range = computeScopeEdgeFieldRange(scopeKeys, fieldName);
    if (!range) return null;
    return { kind: 'edge-field', fieldName, range };
  }
  if (rule.property.startsWith('face-field:')) {
    const fieldName = rule.property.slice(11);
    const range = computeScopeFaceFieldRange(scopeKeys, fieldName);
    if (!range) return null;
    return { kind: 'face-field', fieldName, range };
  }
  return null;
}

function sliderToFilterFrac(value, fallback = 0.0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return clampUnit(numeric / FILTER_SLIDER_MAX, fallback);
}

function filterFracToSlider(frac, fallback = 0.0) {
  return Math.round(clampUnit(frac, fallback) * FILTER_SLIDER_MAX);
}

function rangeFromClip(range, clipState) {
  if (!range) return null;
  return applyClipToRange(range, clipState);
}

function isClipStateRestrictive(clipState) {
  if (!clipState) return false;
  return clampUnit(clipState.minFrac, 0.0) > 0.0
    || clampUnit(clipState.maxFrac, 1.0) < 1.0;
}

function isFilterRuleRestrictive(rule, domain) {
  if (!rule || !domain) return false;
  if (domain.kind === FILTER_POSITION) {
    return ['x', 'y', 'z'].some((axis) => (
      isClipStateRestrictive(rule.positionClip?.[axis])
    ));
  }
  return isClipStateRestrictive(rule.scalarClip);
}

function createDefaultRule(property) {
  return {
    id: `rule-${filterRuleCounter += 1}`,
    scopeKeys: [FILTER_SCOPE_ALL_KEY],
    property: property || FILTER_POSITION,
    expanded: true,
    scalarClip: createClipState(),
    positionClip: {
      x: createClipState(),
      y: createClipState(),
      z: createClipState(),
    },
  };
}

function resetRuleConfig(rule) {
  if (!rule) return;
  rule.scopeKeys = [FILTER_SCOPE_ALL_KEY];
  rule.scalarClip = createClipState();
  rule.positionClip = {
    x: createClipState(),
    y: createClipState(),
    z: createClipState(),
  };
}

function collectMaskIndices(mask, keepPassed) {
  const list = [];
  for (let idx = 0; idx < mask.length; idx += 1) {
    if (Boolean(mask[idx]) === Boolean(keepPassed)) {
      list.push(idx);
    }
  }
  return list;
}

function buildPointSubsetData(entry, sourceIndices, useDefaultColor = false) {
  const basePositions = getEntryPositions(entry);
  const baseColors = entry.currentColors || decodeArraySpec(entry.layer.colors, Uint8Array) || new Uint8Array(0);
  const positions = new Float32Array(sourceIndices.length * 3);
  const colors = new Uint8Array(sourceIndices.length * 3);
  for (let outIdx = 0; outIdx < sourceIndices.length; outIdx += 1) {
    const srcIdx = sourceIndices[outIdx];
    const srcBase = srcIdx * 3;
    const dstBase = outIdx * 3;
    positions[dstBase] = basePositions[srcBase];
    positions[dstBase + 1] = basePositions[srcBase + 1];
    positions[dstBase + 2] = basePositions[srcBase + 2];
    if (useDefaultColor) {
      colors[dstBase] = POINT_CLOUD_DEFAULT_COLOR[0];
      colors[dstBase + 1] = POINT_CLOUD_DEFAULT_COLOR[1];
      colors[dstBase + 2] = POINT_CLOUD_DEFAULT_COLOR[2];
    } else {
      colors[dstBase] = baseColors[srcBase];
      colors[dstBase + 1] = baseColors[srcBase + 1];
      colors[dstBase + 2] = baseColors[srcBase + 2];
    }
  }
  return { positions, colors };
}

function setPointObjectSubset(entry, object3d, sourceIndices, kind = 'unmasked', useDefaultColor = false) {
  if (!object3d) return;
  const geometry = object3d.geometry;
  const subset = buildPointSubsetData(entry, sourceIndices, useDefaultColor);
  setPositionAttribute(geometry, subset.positions);
  setStoredColorAttribute(geometry, subset.colors);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  object3d.userData.maskKind = kind;
}

function ensureMeshVertexPoints(entry) {
  if (entry.layerKind !== 'mesh') return;
  if (!entry.vertexPoints || !entry.maskedVertexPoints) {
    const built = buildMeshVertexPoints(entry.layer, entry.pointSize);
    entry.vertexPoints = built.points;
    entry.maskedVertexPoints = built.maskedPoints;
  }
  if (entry.passMask) {
    updateMeshVertexPointSubsets(entry, entry.passMask);
  } else {
    const allIndices = Array.from({ length: getEntryElementCount(entry) }, (_, idx) => idx);
    setPointObjectSubset(entry, entry.vertexPoints, allIndices, 'unmasked', true);
    setPointObjectSubset(entry, entry.maskedVertexPoints, [], 'masked', true);
    entry.unmaskedPointCount = allIndices.length;
    entry.maskedPointCount = 0;
  }
  updateEntryPointSize(entry, entry.pointSize);
  updateEntryOpacity(entry, entry.opacity);
  updateEntryVertexOverlay(entry);
}

function updateMeshVertexPointSubsets(entry, passMask) {
  if (!entry.vertexPoints || !entry.maskedVertexPoints) return;
  const unmaskedIndices = collectMaskIndices(passMask, true);
  const maskedIndices = collectMaskIndices(passMask, false);
  setPointObjectSubset(entry, entry.vertexPoints, unmaskedIndices, 'unmasked', true);
  setPointObjectSubset(entry, entry.maskedVertexPoints, maskedIndices, 'masked', true);
  entry.unmaskedPointCount = unmaskedIndices.length;
  entry.maskedPointCount = maskedIndices.length;
}

function splitMeshFacesByMask(
  faces,
  vertexPassMask = null,
  faceSize = 3,
  facePassMask = null,
  { includeMasked = true, includeFaceIndices = true, includeFaces = true } = {},
) {
  const width = Math.max(3, Math.min(4, Math.floor(Number(faceSize || 3))));
  const faceCount = Math.floor((faces?.length || 0) / width);
  let unmaskedCount = 0;
  for (let faceIdx = 0; faceIdx < faceCount; faceIdx += 1) {
    const base = faceIdx * width;
    let pass = !facePassMask || Boolean(facePassMask[faceIdx]);
    if (vertexPassMask) {
      for (let local = 0; local < width && pass; local += 1) {
        pass = Boolean(vertexPassMask[faces[base + local]]);
      }
    }
    if (pass) unmaskedCount += 1;
  }
  const maskedCount = includeMasked ? faceCount - unmaskedCount : 0;
  const unmaskedFaces = new Uint32Array(includeFaces ? unmaskedCount * width : 0);
  const maskedFaces = new Uint32Array(includeFaces ? maskedCount * width : 0);
  const unmaskedFaceIndices = includeFaceIndices
    ? new Uint32Array(unmaskedCount)
    : null;
  const maskedFaceIndices = includeFaceIndices
    ? new Uint32Array(maskedCount)
    : null;
  let unmaskedFaceOffset = 0;
  let maskedFaceOffset = 0;
  for (let idx = 0, faceIdx = 0; idx + width - 1 < faces.length; idx += width, faceIdx += 1) {
    let pass = !facePassMask || Boolean(facePassMask[faceIdx]);
    if (vertexPassMask) {
      for (let local = 0; local < width && pass; local += 1) {
        pass = Boolean(vertexPassMask[faces[idx + local]]);
      }
    }
    if (!pass && !includeMasked) continue;
    const targetFaceOffset = pass ? unmaskedFaceOffset : maskedFaceOffset;
    if (includeFaces) {
      const target = pass ? unmaskedFaces : maskedFaces;
      const targetBase = targetFaceOffset * width;
      for (let local = 0; local < width; local += 1) {
        target[targetBase + local] = faces[idx + local];
      }
    }
    const targetIndices = pass ? unmaskedFaceIndices : maskedFaceIndices;
    if (targetIndices) targetIndices[targetFaceOffset] = faceIdx;
    if (pass) unmaskedFaceOffset += 1;
    else maskedFaceOffset += 1;
  }
  return {
    unmaskedFaces,
    maskedFaces,
    unmaskedFaceIndices,
    maskedFaceIndices,
    unmaskedFaceCount: unmaskedCount,
    maskedFaceCount: maskedCount,
  };
}

function buildFlatFaceIndexBuffer(faceIndices, faceSize = 3) {
  const width = Math.max(3, Math.min(4, Math.floor(Number(faceSize || 3))));
  const verticesPerFace = width === 4 ? 6 : 3;
  const indices = new Uint32Array((faceIndices?.length || 0) * verticesPerFace);
  let writeIdx = 0;
  for (let subsetIdx = 0; subsetIdx < (faceIndices?.length || 0); subsetIdx += 1) {
    const base = faceIndices[subsetIdx] * verticesPerFace;
    for (let local = 0; local < verticesPerFace; local += 1) {
      indices[writeIdx] = base + local;
      writeIdx += 1;
    }
  }
  return indices;
}

function ensureFullFlatFaceGeometry(entry) {
  const faceSize = meshFaceSize(entry.layer);
  const verticesPerFace = faceSize === 4 ? 6 : 3;
  const faceCount = Math.floor(entry.layer.faces.length / faceSize);
  const position = entry.mesh.geometry.getAttribute('position');
  if (!position || position.count !== faceCount * verticesPerFace) {
    setFlatFaceGeometry(
      entry.mesh,
      entry.layer,
      null,
      entry.currentFaceColors,
      { refreshBounds: true },
    );
  } else if (entry.mesh.userData.flatFaceColorsSource !== entry.currentFaceColors) {
    // Positions stay resident across face-field/cmap changes; only the
    // expanded per-corner color buffer needs replacing.
    setStoredColorAttribute(
      entry.mesh.geometry,
      expandFlatFaceColors(entry.currentFaceColors, faceCount, verticesPerFace),
    );
    entry.mesh.userData.flatFaceColorsSource = entry.currentFaceColors;
  }
}

function expandFlatFaceColors(faceColors, faceCount, verticesPerFace) {
  const colors = new Uint8Array(faceCount * verticesPerFace * 3);
  for (let faceIdx = 0; faceIdx < faceCount; faceIdx += 1) {
    const source = faceIdx * 3;
    const red = faceColors[source];
    const green = faceColors[source + 1];
    const blue = faceColors[source + 2];
    let target = faceIdx * verticesPerFace * 3;
    for (let corner = 0; corner < verticesPerFace; corner += 1) {
      colors[target] = red;
      colors[target + 1] = green;
      colors[target + 2] = blue;
      target += 3;
    }
  }
  return colors;
}

function buildFlatFaceSubset(layer, faceIndices, faceColors) {
  const width = meshFaceSize(layer);
  const verticesPerFace = width === 4 ? 6 : 3;
  const faceCount = faceIndices
    ? faceIndices.length
    : Math.floor(layer.faces.length / width);
  const positions = new Float32Array(faceCount * verticesPerFace * 3);
  const colors = new Uint8Array(faceCount * verticesPerFace * 3);
  const localOrder = width === 4 ? [0, 1, 2, 0, 2, 3] : [0, 1, 2];
  let writeVertex = 0;
  for (let subsetIdx = 0; subsetIdx < faceCount; subsetIdx += 1) {
    const faceIdx = faceIndices ? faceIndices[subsetIdx] : subsetIdx;
    const colorBase = faceIdx * 3;
    for (const localIdx of localOrder) {
      const sourceVertex = layer.faces[faceIdx * width + localIdx];
      const sourceBase = sourceVertex * 3;
      const targetBase = writeVertex * 3;
      positions[targetBase] = layer.vertices[sourceBase];
      positions[targetBase + 1] = layer.vertices[sourceBase + 1];
      positions[targetBase + 2] = layer.vertices[sourceBase + 2];
      colors[targetBase] = faceColors[colorBase];
      colors[targetBase + 1] = faceColors[colorBase + 1];
      colors[targetBase + 2] = faceColors[colorBase + 2];
      writeVertex += 1;
    }
  }
  return { positions, colors };
}

function setFlatFaceGeometry(
  object3d,
  layer,
  faceIndices,
  faceColors,
  { refreshBounds = true } = {},
) {
  const subset = buildFlatFaceSubset(layer, faceIndices, faceColors);
  setPositionAttribute(object3d.geometry, subset.positions);
  setStoredColorAttribute(object3d.geometry, subset.colors);
  object3d.geometry.setIndex(null);
  object3d.geometry.deleteAttribute('uv');
  object3d.geometry.deleteAttribute('normal');
  object3d.userData.flatFaceColorsSource = faceIndices ? null : faceColors;
  if (refreshBounds) refreshMeshBounds(object3d.geometry, false);
}

function restoreIndexedMeshGeometry(entry, colors) {
  setPositionAttribute(entry.mesh.geometry, entry.layer.vertices);
  setStoredColorAttribute(entry.mesh.geometry, colors);
  entry.mesh.userData.flatFaceColorsSource = null;
  if (entry.layer.texture?.uv) {
    entry.mesh.geometry.setAttribute('uv', new THREE.BufferAttribute(entry.layer.texture.uv, 2));
  }
}

function updateMeshEdgeLinesFromGeometry(entry) {
  if (!entry.edgeLines || !entry.mesh) return;
  const useMeshEdgeDomain = shouldUseMeshEdgeDomain(entry);
  if (Boolean(entry.edgeLines.userData.meshEdgeDomain) !== useMeshEdgeDomain) {
    rebuildMeshEdgeOverlay(entry);
    return;
  }
  if (useMeshEdgeDomain) {
    const edgeColors = entry.activeFieldAssociation === 'edge'
      ? entry.currentEdgeColors
      : null;
    const segmentData = buildPointEdgeSegmentData(
      entry.layer,
      entry.filterActive ? entry.passMask : null,
      entry.edgePassMask || null,
      edgeColors,
    );
    entry.edgeLines.geometry.setPositions(segmentData.positions);
    if (segmentData.colors) entry.edgeLines.geometry.setColors(segmentData.colors);
    entry.edgeLines.geometry.instanceCount = segmentData.edgeCount;
    entry.edgeLines.material.vertexColors = Boolean(segmentData.colors);
    entry.edgeLines.material.needsUpdate = true;
    entry.edgeLines.userData.edgeCount = segmentData.edgeCount;
    entry.unmaskedEdgeCount = segmentData.edgeCount;
    return;
  }
  rebuildMeshEdgeOverlay(entry);
}

function buildPointEdgeSegmentData(layer, pointPassMask = null, edgePassMask = null, edgeColors = null) {
  const baseEdges = layer.edges;
  const basePositions = layer.points?.length ? layer.points : layer.vertices;
  const sourceEdgeCount = Math.floor((baseEdges?.length || 0) / 2);
  if (!baseEdges || sourceEdgeCount === 0) {
    return { positions: new Float32Array(0), colors: null, edgeCount: 0 };
  }

  let edgeCount = 0;
  for (let edgeIdx = 0; edgeIdx < sourceEdgeCount; edgeIdx += 1) {
    if (edgePassMask && !edgePassMask[edgeIdx]) continue;
    const i = baseEdges[edgeIdx * 2];
    const j = baseEdges[edgeIdx * 2 + 1];
    if (pointPassMask && (!pointPassMask[i] || !pointPassMask[j])) continue;
    const iBase = i * 3;
    const jBase = j * 3;
    if (iBase + 2 >= basePositions.length || jBase + 2 >= basePositions.length) continue;
    edgeCount += 1;
  }

  const positions = new Float32Array(edgeCount * 6);
  const hasEdgeColors = Boolean(edgeColors && edgeColors.length >= sourceEdgeCount * 3);
  const colors = hasEdgeColors ? new Float32Array(edgeCount * 6) : null;
  let writeBase = 0;
  for (let edgeIdx = 0; edgeIdx < sourceEdgeCount; edgeIdx += 1) {
    if (edgePassMask && !edgePassMask[edgeIdx]) continue;
    const i = baseEdges[edgeIdx * 2];
    const j = baseEdges[edgeIdx * 2 + 1];
    if (pointPassMask && (!pointPassMask[i] || !pointPassMask[j])) continue;
    const iBase = i * 3;
    const jBase = j * 3;
    if (iBase + 2 >= basePositions.length || jBase + 2 >= basePositions.length) continue;
    positions[writeBase] = basePositions[iBase];
    positions[writeBase + 1] = basePositions[iBase + 1];
    positions[writeBase + 2] = basePositions[iBase + 2];
    positions[writeBase + 3] = basePositions[jBase];
    positions[writeBase + 4] = basePositions[jBase + 1];
    positions[writeBase + 5] = basePositions[jBase + 2];
    if (colors) {
      const colorBase = edgeIdx * 3;
      const r = edgeColors[colorBase] / 255;
      const g = edgeColors[colorBase + 1] / 255;
      const b = edgeColors[colorBase + 2] / 255;
      colors[writeBase] = r;
      colors[writeBase + 1] = g;
      colors[writeBase + 2] = b;
      colors[writeBase + 3] = r;
      colors[writeBase + 4] = g;
      colors[writeBase + 5] = b;
    }
    writeBase += 6;
  }
  return { positions, colors, edgeCount };
}

function updatePointLinesFromMask(entry, passMask, edgePassMask = null) {
  if (!entry.pointLines) return;
  // Chunked indexed graphs deliberately retain complete connectivity while
  // point filters are active; rebuilding their full edge domain would defeat
  // the bounded-allocation path.
  if (entry.pointLines.userData.indexedPointEdgeChunks) return;
  const geometry = entry.pointLines.geometry;
  const positionData = geometry.getAttribute('instanceStart')?.data || null;
  const colorData = geometry.getAttribute('instanceColorStart')?.data || null;
  const positions = positionData?.array || null;
  const colors = colorData?.array || null;
  const baseEdges = entry.layer.edges;
  const basePositions = entry.layer.points;
  const edgeColors = entry.currentEdgeColors || null;
  const sourceEdgeCount = Math.floor((baseEdges?.length || 0) / 2);
  let edgeCount = 0;
  for (let edgeIdx = 0; edgeIdx < sourceEdgeCount; edgeIdx += 1) {
    if (edgePassMask && !edgePassMask[edgeIdx]) continue;
    const i = baseEdges[edgeIdx * 2];
    const j = baseEdges[edgeIdx * 2 + 1];
    if (passMask && (!passMask[i] || !passMask[j])) continue;
    const iBase = i * 3;
    const jBase = j * 3;
    const writeBase = edgeCount * 6;
    if (
      iBase + 2 >= basePositions.length
      || jBase + 2 >= basePositions.length
      || !positions
      || writeBase + 5 >= positions.length
    ) continue;
    positions[writeBase] = basePositions[iBase];
    positions[writeBase + 1] = basePositions[iBase + 1];
    positions[writeBase + 2] = basePositions[iBase + 2];
    positions[writeBase + 3] = basePositions[jBase];
    positions[writeBase + 4] = basePositions[jBase + 1];
    positions[writeBase + 5] = basePositions[jBase + 2];
    if (colors && edgeColors && edgeColors.length >= (edgeIdx + 1) * 3) {
      const colorBase = edgeIdx * 3;
      const r = edgeColors[colorBase] / 255;
      const g = edgeColors[colorBase + 1] / 255;
      const b = edgeColors[colorBase + 2] / 255;
      colors[writeBase] = r;
      colors[writeBase + 1] = g;
      colors[writeBase + 2] = b;
      colors[writeBase + 3] = r;
      colors[writeBase + 4] = g;
      colors[writeBase + 5] = b;
    }
    edgeCount += 1;
  }
  if (positionData) positionData.needsUpdate = true;
  if (colorData) colorData.needsUpdate = true;
  geometry.instanceCount = edgeCount;
  entry.pointLines.userData.edgeCount = edgeCount;
  entry.unmaskedEdgeCount = edgeCount;
}

function setLayerEdgeField(entry, fieldIdx) {
  const fieldCount = getEntryFieldCount(entry, 'edge');
  if (fieldCount <= 0) return;
  const nextIdx = Math.max(0, Math.min(fieldCount - 1, fieldIdx));
  renderEntryColors(entry, nextIdx, 'edge');
  updateColorbars();
}

function evaluateFieldRuleOnMask(entry, mask, rule, domain) {
  const fieldName = domain.fieldName;
  const values = getEntryFieldValuesByName(entry, fieldName);
  if (!values) return;
  const activeRange = rangeFromClip(domain.range, rule.scalarClip || createClipState());
  if (!activeRange) return;
  for (let idx = 0; idx < values.length; idx += 1) {
    if (!mask[idx]) continue;
    const value = Number(values[idx]);
    if (!Number.isFinite(value) || value < activeRange.min || value > activeRange.max) {
      mask[idx] = 0;
    }
  }
}

function evaluatePositionRuleOnMask(entry, mask, rule, domain) {
  const positions = getEntryPositions(entry);
  if (!positions || positions.length === 0) return;
  const ranges = domain.ranges;
  const xRange = rangeFromClip(ranges.x, rule.positionClip?.x || createClipState());
  const yRange = rangeFromClip(ranges.y, rule.positionClip?.y || createClipState());
  const zRange = rangeFromClip(ranges.z, rule.positionClip?.z || createClipState());
  if (!xRange || !yRange || !zRange) return;
  const count = getEntryElementCount(entry);
  for (let idx = 0; idx < count; idx += 1) {
    if (!mask[idx]) continue;
    const base = idx * 3;
    const x = Number(positions[base]);
    const y = Number(positions[base + 1]);
    const z = Number(positions[base + 2]);
    const valid = Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)
      && x >= xRange.min && x <= xRange.max
      && y >= yRange.min && y <= yRange.max
      && z >= zRange.min && z <= zRange.max;
    if (!valid) {
      mask[idx] = 0;
    }
  }
}

function computeEntryPassMask(entry) {
  const count = getEntryElementCount(entry);
  const mask = new Uint8Array(count);
  mask.fill(1);
  if (!filterState.rules || filterState.rules.length === 0) {
    return mask;
  }
  for (const rule of filterState.rules) {
    const ruleScopeKeys = resolveScopeKeys(rule.scopeKeys, false);
    if (ruleScopeKeys.length === 0) continue;
    if (!isEntryWithinScope(entry, ruleScopeKeys, false)) continue;
    const domain = getRuleDomain(rule, ruleScopeKeys);
    if (!domain || !isFilterRuleRestrictive(rule, domain)) continue;
    if (domain.kind === FILTER_POSITION) {
      evaluatePositionRuleOnMask(entry, mask, rule, domain);
    } else if (domain.kind === 'field') {
      evaluateFieldRuleOnMask(entry, mask, rule, domain);
    }
  }
  return mask;
}

function computeEntryEdgePassMask(entry) {
  const count = Math.floor((entry.layer.edges?.length || 0) / 2);
  const mask = new Uint8Array(count);
  mask.fill(1);
  for (const rule of filterState.rules || []) {
    const ruleScopeKeys = resolveScopeKeys(rule.scopeKeys, false);
    if (ruleScopeKeys.length === 0 || !isEntryWithinScope(entry, ruleScopeKeys, false)) continue;
    const domain = getRuleDomain(rule, ruleScopeKeys);
    if (
      !domain
      || domain.kind !== 'edge-field'
      || !isFilterRuleRestrictive(rule, domain)
    ) continue;
    const values = getEntryEdgeFieldValuesByName(entry, domain.fieldName);
    const activeRange = rangeFromClip(domain.range, rule.scalarClip || createClipState());
    if (!values || !activeRange) continue;
    for (let idx = 0; idx < Math.min(values.length, mask.length); idx += 1) {
      const value = Number(values[idx]);
      if (!Number.isFinite(value) || value < activeRange.min || value > activeRange.max) mask[idx] = 0;
    }
  }
  return mask;
}

function computeEntryFacePassMask(entry) {
  const count = Math.floor((entry.layer.faces?.length || 0) / meshFaceSize(entry.layer));
  const mask = new Uint8Array(count);
  mask.fill(1);
  for (const rule of filterState.rules || []) {
    const ruleScopeKeys = resolveScopeKeys(rule.scopeKeys, false);
    if (ruleScopeKeys.length === 0 || !isEntryWithinScope(entry, ruleScopeKeys, false)) continue;
    const domain = getRuleDomain(rule, ruleScopeKeys);
    if (
      !domain
      || domain.kind !== 'face-field'
      || !isFilterRuleRestrictive(rule, domain)
    ) continue;
    const values = getEntryFaceFieldValuesByName(entry, domain.fieldName);
    const activeRange = rangeFromClip(domain.range, rule.scalarClip || createClipState());
    if (!values || !activeRange) continue;
    for (let idx = 0; idx < Math.min(values.length, mask.length); idx += 1) {
      const value = Number(values[idx]);
      if (!Number.isFinite(value) || value < activeRange.min || value > activeRange.max) mask[idx] = 0;
    }
  }
  return mask;
}

function computeEntryEdgeFacePassMask(entry, edgePassMask) {
  const faceEdges = entry.layer.face_edges;
  if (!edgePassMask || !faceEdges?.length) return null;
  const faceSize = meshFaceSize(entry.layer);
  const faceCount = Math.floor(entry.layer.faces.length / faceSize);
  if (faceEdges.length < faceCount * faceSize) return null;
  const mask = new Uint8Array(faceCount);
  mask.fill(1);
  const missingEdge = 0xffffffff;
  for (let faceIdx = 0; faceIdx < faceCount; faceIdx += 1) {
    const base = faceIdx * faceSize;
    for (let localIdx = 0; localIdx < faceSize; localIdx += 1) {
      const edgeIdx = faceEdges[base + localIdx];
      if (edgeIdx !== missingEdge && !edgePassMask[edgeIdx]) {
        mask[faceIdx] = 0;
        break;
      }
    }
  }
  return mask;
}

function intersectPassMasks(first, second) {
  if (!first) return second;
  if (!second) return first;
  const count = Math.min(first.length, second.length);
  for (let idx = 0; idx < count; idx += 1) {
    first[idx] = first[idx] && second[idx] ? 1 : 0;
  }
  return first;
}

function getEntryFilterActivity(entry) {
  let point = false;
  let edge = false;
  let face = false;
  for (const rule of filterState.rules || []) {
    const ruleScopeKeys = resolveScopeKeys(rule.scopeKeys, false);
    if (ruleScopeKeys.length === 0 || !isEntryWithinScope(entry, ruleScopeKeys, false)) continue;
    const domain = getRuleDomain(rule, ruleScopeKeys);
    if (!domain || !isFilterRuleRestrictive(rule, domain)) continue;
    if (domain.kind === FILTER_POSITION) {
      point = true;
    } else if (domain.kind === 'field' && getEntryFieldValuesByName(entry, domain.fieldName)) {
      point = true;
    } else if (
      domain.kind === 'edge-field'
      && getEntryEdgeFieldValuesByName(entry, domain.fieldName)
    ) {
      edge = true;
    } else if (
      domain.kind === 'face-field'
      && getEntryFaceFieldValuesByName(entry, domain.fieldName)
    ) {
      face = true;
    }
  }
  return { point, edge, face };
}

function lowerBoundSorted(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = low + ((high - low) >> 1);
    if (values[mid] < target) low = mid + 1;
    else high = mid;
  }
  return low;
}

function upperBoundSorted(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = low + ((high - low) >> 1);
    if (values[mid] <= target) low = mid + 1;
    else high = mid;
  }
  return low;
}

function canUseSortedFaceFilterWithEdges(entry) {
  if (!shouldDisplayMeshEdges(entry)) return true;
  return !shouldUseMeshEdgeDomain(entry) && entry.edgeThickness <= 1.0;
}

function applySortedMeshEdgeDrawRange(entry, begin, end) {
  if (!shouldDisplayMeshEdges(entry)) return true;
  if (!canUseSortedFaceFilterWithEdges(entry)) return false;
  if (entry.edgeLines && !entry.edgeLines.userData.contiguousFullTopology) {
    disposeMeshEdgeOverlay(entry);
  }
  if (!entry.edgeLines) entry.edgeLines = buildMeshEdgeLines(entry);
  const group = entry.edgeLines;
  if (!group || group.userData.wideEdges || group.userData.meshEdgeDomain) return false;

  let edgeCount = 0;
  for (const lines of group.children) {
    const faceBegin = Number(lines.userData.sourceFaceBegin);
    const faceEnd = Number(lines.userData.sourceFaceEnd);
    const segmentsPerFace = Number(lines.userData.segmentsPerFace);
    if (!Number.isFinite(faceBegin) || !Number.isFinite(faceEnd) || !segmentsPerFace) {
      return false;
    }
    const overlapBegin = Math.max(begin, faceBegin);
    const overlapEnd = Math.min(end, faceEnd);
    const selectedFaceCount = Math.max(0, overlapEnd - overlapBegin);
    const firstIndex = (overlapBegin - faceBegin) * segmentsPerFace * 2;
    const indexCount = selectedFaceCount * segmentsPerFace * 2;
    lines.geometry.setDrawRange(firstIndex, indexCount);
    lines.visible = selectedFaceCount > 0;
    edgeCount += selectedFaceCount * segmentsPerFace;
  }
  group.userData.edgeCount = edgeCount;
  entry.unmaskedEdgeCount = edgeCount;
  return true;
}

function resetSortedMeshEdgeDrawRange(entry) {
  const group = entry.edgeLines;
  if (!group || group.userData.wideEdges || group.userData.meshEdgeDomain
      || !group.userData.contiguousFullTopology) return false;
  let edgeCount = 0;
  for (const lines of group.children) {
    const faceBegin = Number(lines.userData.sourceFaceBegin);
    const faceEnd = Number(lines.userData.sourceFaceEnd);
    const segmentsPerFace = Number(lines.userData.segmentsPerFace);
    if (!Number.isFinite(faceBegin) || !Number.isFinite(faceEnd) || !segmentsPerFace) {
      return false;
    }
    const chunkEdgeCount = (faceEnd - faceBegin) * segmentsPerFace;
    lines.geometry.setDrawRange(0, chunkEdgeCount * 2);
    lines.visible = chunkEdgeCount > 0;
    edgeCount += chunkEdgeCount;
  }
  group.userData.edgeCount = edgeCount;
  entry.unmaskedEdgeCount = edgeCount;
  return true;
}

function getSortedFaceFilterSpec(entry) {
  if (
    entry.layerKind !== 'mesh'
    || filterState.maskedOpacity > 0.0
    || entry.showVertices
    || !canUseSortedFaceFilterWithEdges(entry)
  ) return null;
  let result = null;
  for (const rule of filterState.rules || []) {
    const ruleScopeKeys = resolveScopeKeys(rule.scopeKeys, false);
    if (ruleScopeKeys.length === 0 || !isEntryWithinScope(entry, ruleScopeKeys, false)) continue;
    const domain = getRuleDomain(rule, ruleScopeKeys);
    if (!domain || !isFilterRuleRestrictive(rule, domain)) continue;
    if (domain.kind !== 'face-field') return null;
    const fieldIdx = (entry.layer.face_field_names || []).findIndex(
      (name) => String(name) === String(domain.fieldName),
    );
    if (fieldIdx < 0) continue;
    if (!entry.layer.face_field_sorted_ascending?.[fieldIdx]) return null;
    const values = entry.layer.face_field_values?.[fieldIdx] || null;
    const activeRange = rangeFromClip(domain.range, rule.scalarClip || createClipState());
    if (!values || !activeRange || result !== null) return null;
    result = { values, activeRange };
  }
  return result;
}

function clearMaskedMeshGeometry(entry) {
  if (!entry.maskedMesh) return;
  setPositionAttribute(entry.maskedMesh.geometry, new Float32Array(0));
  setStoredColorAttribute(entry.maskedMesh.geometry, new Uint8Array(0));
  setIndexAttribute(entry.maskedMesh.geometry, new Uint32Array(0));
  entry.maskedMesh.geometry.deleteAttribute('normal');
  entry.maskedMesh.geometry.deleteAttribute('uv');
}

function applySortedFaceFilter(entry, spec, colors, usesFaceColors) {
  const faceSize = meshFaceSize(entry.layer);
  const faceCount = Math.floor(entry.layer.faces.length / faceSize);
  if (spec.values.length !== faceCount) return false;
  const begin = lowerBoundSorted(spec.values, spec.activeRange.min);
  const end = upperBoundSorted(spec.values, spec.activeRange.max);
  if (usesFaceColors) {
    setMeshFaceColorMode(entry, true);
    ensureFullFlatFaceGeometry(entry);
    // The flat buffer is already in sorted source-face order. A non-indexed
    // draw range selects the desired contiguous score interval directly.
    entry.mesh.geometry.setIndex(null);
    const verticesPerFace = faceSize === 4 ? 6 : 3;
    entry.mesh.geometry.setDrawRange(
      begin * verticesPerFace,
      (end - begin) * verticesPerFace,
    );
  } else {
    const needsIndexedRestore = (
      (entry.filterActive && !entry.sortedFaceFilterActive)
      || entry.sortedFaceFilterUsesFaceColors
    );
    if (needsIndexedRestore) {
      restoreIndexedMeshGeometry(entry, colors);
      setIndexAttribute(entry.mesh.geometry, triangulateFaces(entry.layer.faces, faceSize));
    }
    const indicesPerFace = faceSize === 4 ? 6 : 3;
    entry.mesh.geometry.setDrawRange(
      begin * indicesPerFace,
      (end - begin) * indicesPerFace,
    );
  }
  clearMaskedMeshGeometry(entry);
  entry.passMask = null;
  entry.edgePassMask = null;
  entry.facePassMask = null;
  entry.filterActive = true;
  entry.pointFilterActive = false;
  entry.edgeFilterActive = false;
  entry.faceFilterActive = true;
  entry.sortedFaceFilterActive = true;
  entry.sortedFaceFilterUsesFaceColors = usesFaceColors;
  entry.sortedFaceFilterBegin = begin;
  entry.sortedFaceFilterEnd = end;
  entry.unmaskedFaceCount = end - begin;
  entry.maskedFaceCount = faceCount - (end - begin);
  updateEntryOpacity(entry, entry.opacity);
  updateEntryEdgeOverlay(entry);
  return true;
}

function applyFiltersToEntry(entry) {
  // Hidden layers are lazy placeholders until first shown.
  if (!entry || !entry.loaded) return;
  const colors = entry.currentColors || decodeArraySpec(entry.layer.colors, Uint8Array) || new Uint8Array(0);
  const activity = getEntryFilterActivity(entry);
  const usesFaceColors = entry.layerKind === 'mesh'
    && entry.activeSurfaceAssociation === 'face'
    && entry.currentFaceColors;
  const sortedFaceFilter = getSortedFaceFilterSpec(entry);
  if (sortedFaceFilter && applySortedFaceFilter(
    entry, sortedFaceFilter, colors, usesFaceColors,
  )) return;

  // The overwhelmingly common display path has no filters. Rendering the
  // original typed arrays directly avoids an O(N) mask, an N-element JS index
  // list, and full copies of point positions/colors (or mesh indices).
  if (!activity.point && !activity.edge && !activity.face) {
    entry.passMask = null;
    entry.edgePassMask = null;
    entry.facePassMask = null;
    if (entry.layerKind === 'pointcloud') {
      if (entry.pointFilterActive) {
        setPositionAttribute(entry.points.geometry, entry.layer.points);
        setStoredColorAttribute(entry.points.geometry, colors);
        entry.points.geometry.computeBoundingBox();
        entry.points.geometry.computeBoundingSphere();
        setPositionAttribute(entry.maskedPoints.geometry, new Float32Array(0));
        setStoredColorAttribute(entry.maskedPoints.geometry, new Uint8Array(0));
      } else {
        setStoredColorAttribute(entry.points.geometry, colors);
      }
      entry.unmaskedPointCount = getEntryElementCount(entry);
      entry.maskedPointCount = 0;
      if (entry.pointLines && entry.filterActive) {
        updatePointLinesFromMask(entry, null, null);
      }
    } else {
      const faceSize = meshFaceSize(entry.layer);
      if (usesFaceColors) {
        setMeshFaceColorMode(entry, true);
        if (entry.sortedFaceFilterActive) {
          ensureFullFlatFaceGeometry(entry);
          entry.mesh.geometry.setIndex(null);
        } else {
          setFlatFaceGeometry(entry.mesh, entry.layer, null, entry.currentFaceColors);
        }
      } else {
        if (!entry.sortedFaceFilterActive || entry.sortedFaceFilterUsesFaceColors) {
          restoreIndexedMeshGeometry(entry, colors);
          setIndexAttribute(entry.mesh.geometry, triangulateFaces(entry.layer.faces, faceSize));
        }
      }
      entry.mesh.geometry.setDrawRange(0, Infinity);
      if (entry.filterActive || usesFaceColors) {
        setPositionAttribute(entry.maskedMesh.geometry, new Float32Array(0));
        setStoredColorAttribute(entry.maskedMesh.geometry, new Uint8Array(0));
        setIndexAttribute(entry.maskedMesh.geometry, new Uint32Array(0));
        entry.maskedMesh.geometry.deleteAttribute('uv');
      }
      entry.unmaskedFaceCount = Math.floor(entry.layer.faces.length / meshFaceSize(entry.layer));
      entry.maskedFaceCount = 0;
      if (entry.vertexPoints || entry.maskedVertexPoints) {
        ensureMeshVertexPoints(entry);
      }
    }
    const wasFiltered = entry.filterActive;
    const wasSortedFaceFilter = Boolean(entry.sortedFaceFilterActive);
    entry.filterActive = false;
    entry.pointFilterActive = false;
    entry.edgeFilterActive = false;
    entry.faceFilterActive = false;
    entry.sortedFaceFilterActive = false;
    entry.sortedFaceFilterUsesFaceColors = false;
    entry.sortedFaceFilterBegin = null;
    entry.sortedFaceFilterEnd = null;
    if (entry.layerKind === 'mesh' && (wasFiltered || entry.layer.edges?.length)) {
      if (!wasSortedFaceFilter || !resetSortedMeshEdgeDrawRange(entry)) {
        updateMeshEdgeLinesFromGeometry(entry);
      }
    }
    updateEntryOpacity(entry, entry.opacity);
    updateEntryEdgeOverlay(entry);
    return;
  }

  const wasPointFilterActive = Boolean(entry.pointFilterActive);
  if (entry.layerKind === 'mesh') entry.mesh.geometry.setDrawRange(0, Infinity);
  entry.sortedFaceFilterActive = false;
  entry.sortedFaceFilterUsesFaceColors = false;
  entry.sortedFaceFilterBegin = null;
  entry.sortedFaceFilterEnd = null;
  const passMask = activity.point ? computeEntryPassMask(entry) : null;
  const edgePassMask = activity.edge ? computeEntryEdgePassMask(entry) : null;
  let facePassMask = activity.face ? computeEntryFacePassMask(entry) : null;
  if (activity.edge && entry.layerKind === 'mesh') {
    facePassMask = intersectPassMasks(
      facePassMask,
      computeEntryEdgeFacePassMask(entry, edgePassMask),
    );
  }
  entry.passMask = passMask;
  entry.edgePassMask = edgePassMask;
  entry.facePassMask = facePassMask;
  entry.filterActive = true;
  entry.pointFilterActive = activity.point;
  entry.edgeFilterActive = activity.edge;
  entry.faceFilterActive = Boolean(facePassMask);

  if (entry.layerKind === 'pointcloud') {
    if (activity.point) {
      const unmaskedIndices = collectMaskIndices(passMask, true);
      const maskedIndices = collectMaskIndices(passMask, false);
      setPointObjectSubset(entry, entry.points, unmaskedIndices, 'unmasked');
      setPointObjectSubset(entry, entry.maskedPoints, maskedIndices, 'masked');
      entry.unmaskedPointCount = unmaskedIndices.length;
      entry.maskedPointCount = maskedIndices.length;
    } else {
      if (wasPointFilterActive) {
        setPositionAttribute(entry.points.geometry, entry.layer.points);
        entry.points.geometry.computeBoundingBox();
        entry.points.geometry.computeBoundingSphere();
        setPositionAttribute(entry.maskedPoints.geometry, new Float32Array(0));
        setStoredColorAttribute(entry.maskedPoints.geometry, new Uint8Array(0));
      }
      setStoredColorAttribute(entry.points.geometry, colors);
      entry.unmaskedPointCount = getEntryElementCount(entry);
      entry.maskedPointCount = 0;
    }
    updatePointLinesFromMask(entry, passMask, edgePassMask);
  } else {
    const faceSize = meshFaceSize(entry.layer);
    // With the default zero masked opacity there is no reason to allocate,
    // populate, and upload the rejected half of a multi-million-face mesh.
    // It is rebuilt on demand if the user later enables masked visibility.
    const includeMasked = filterState.maskedOpacity > 0.0;
    const splitFaces = splitMeshFacesByMask(
      entry.layer.faces,
      passMask,
      faceSize,
      facePassMask,
      {
        includeMasked,
        includeFaceIndices: usesFaceColors,
        includeFaces: !usesFaceColors,
      },
    );
    if (usesFaceColors) {
      setMeshFaceColorMode(entry, true);
      // Face coloring already expands each source face into independent flat
      // vertices. Keep that large position/color buffer resident and filter by
      // replacing only its index buffer; rebuilding the flat attributes here
      // copied hundreds of MB for large meshes on every threshold change.
      ensureFullFlatFaceGeometry(entry);
      setIndexAttribute(
        entry.mesh.geometry,
        buildFlatFaceIndexBuffer(splitFaces.unmaskedFaceIndices, faceSize),
      );
      if (includeMasked) {
        entry.maskedMesh.geometry.setAttribute(
          'position', entry.mesh.geometry.getAttribute('position'),
        );
        entry.maskedMesh.geometry.setAttribute(
          'color', entry.mesh.geometry.getAttribute('color'),
        );
        entry.maskedMesh.geometry.boundingBox = entry.mesh.geometry.boundingBox;
        entry.maskedMesh.geometry.boundingSphere = entry.mesh.geometry.boundingSphere;
        setIndexAttribute(
          entry.maskedMesh.geometry,
          buildFlatFaceIndexBuffer(splitFaces.maskedFaceIndices, faceSize),
        );
      } else {
        setPositionAttribute(entry.maskedMesh.geometry, new Float32Array(0));
        setStoredColorAttribute(entry.maskedMesh.geometry, new Uint8Array(0));
        entry.maskedMesh.geometry.setIndex(null);
        entry.maskedMesh.geometry.deleteAttribute('uv');
      }
    } else {
      restoreIndexedMeshGeometry(entry, colors);
      setIndexAttribute(entry.mesh.geometry, triangulateFaces(splitFaces.unmaskedFaces, faceSize));
      // Vertex positions, normals, and full-mesh bounds are unchanged by an
      // index-only filter. Reuse them instead of recomputing normals/bounds or
      // uploading a duplicate vertex buffer for the masked mesh.
      if (includeMasked) {
        entry.maskedMesh.geometry.setAttribute(
          'position', entry.mesh.geometry.getAttribute('position'),
        );
        entry.maskedMesh.geometry.setAttribute(
          'color', entry.mesh.geometry.getAttribute('color'),
        );
        const normals = entry.mesh.geometry.getAttribute('normal');
        if (normals) entry.maskedMesh.geometry.setAttribute('normal', normals);
        if (entry.layer.texture?.uv) {
          entry.maskedMesh.geometry.setAttribute(
            'uv', entry.mesh.geometry.getAttribute('uv'),
          );
        }
        entry.maskedMesh.geometry.boundingBox = entry.mesh.geometry.boundingBox;
        entry.maskedMesh.geometry.boundingSphere = entry.mesh.geometry.boundingSphere;
        setIndexAttribute(
          entry.maskedMesh.geometry,
          triangulateFaces(splitFaces.maskedFaces, faceSize),
        );
      } else {
        setPositionAttribute(entry.maskedMesh.geometry, new Float32Array(0));
        setStoredColorAttribute(entry.maskedMesh.geometry, new Uint8Array(0));
        setIndexAttribute(entry.maskedMesh.geometry, new Uint32Array(0));
        entry.maskedMesh.geometry.deleteAttribute('normal');
        entry.maskedMesh.geometry.deleteAttribute('uv');
      }
    }
    entry.unmaskedFaceCount = splitFaces.unmaskedFaceCount;
    entry.maskedFaceCount = splitFaces.maskedFaceCount;
    if (passMask) {
      updateMeshVertexPointSubsets(entry, passMask);
    } else if (wasPointFilterActive && (entry.vertexPoints || entry.maskedVertexPoints)) {
      entry.passMask = null;
      ensureMeshVertexPoints(entry);
    }
    updateMeshEdgeLinesFromGeometry(entry);
  }

  updateEntryOpacity(entry, entry.opacity);
  updateEntryEdgeOverlay(entry);
}

function applyFiltersToAllEntries() {
  for (const entry of layers) {
    if (!entry.loaded) continue;
    applyFiltersToEntry(entry);
  }
  requestRender();
}

function requestApplyFiltersToAllEntries() {
  if (filterApplyQueued) return;
  filterApplyQueued = true;
  window.requestAnimationFrame(() => {
    filterApplyQueued = false;
    applyFiltersToAllEntries();
  });
}

function formatFilterValue(value) {
  return Number.isFinite(value) ? Number(value).toFixed(4) : 'n/a';
}

function createDualSliderControl({
  clipState,
  range,
  label,
  onChanged,
}) {
  const wrap = document.createElement('div');
  wrap.className = 'filter-control';
  const title = document.createElement('label');
  title.textContent = label;

  const valuesLabel = document.createElement('div');
  valuesLabel.className = 'filter-dual-label';
  const left = document.createElement('span');
  const right = document.createElement('span');
  valuesLabel.appendChild(left);
  valuesLabel.appendChild(right);

  const slider = document.createElement('div');
  slider.className = 'filter-dual-slider';
  const track = document.createElement('div');
  track.className = 'filter-dual-track';
  const fill = document.createElement('div');
  fill.className = 'filter-dual-fill';
  const minInput = document.createElement('input');
  minInput.type = 'range';
  minInput.min = '0';
  minInput.max = String(FILTER_SLIDER_MAX);
  minInput.step = '1';
  minInput.value = String(filterFracToSlider(clipState.minFrac, 0.0));
  const maxInput = document.createElement('input');
  maxInput.type = 'range';
  maxInput.min = '0';
  maxInput.max = String(FILTER_SLIDER_MAX);
  maxInput.step = '1';
  maxInput.value = String(filterFracToSlider(clipState.maxFrac, 1.0));

  slider.appendChild(track);
  slider.appendChild(fill);
  slider.appendChild(minInput);
  slider.appendChild(maxInput);

  const sync = () => {
    minInput.value = String(filterFracToSlider(clipState.minFrac, 0.0));
    maxInput.value = String(filterFracToSlider(clipState.maxFrac, 1.0));
    const startPct = clampUnit(clipState.minFrac, 0.0) * 100.0;
    const endPct = clampUnit(clipState.maxFrac, 1.0) * 100.0;
    fill.style.left = `${startPct}%`;
    fill.style.width = `${Math.max(0.0, endPct - startPct)}%`;
    const clippedRange = rangeFromClip(range, clipState);
    left.textContent = formatFilterValue(clippedRange?.min);
    right.textContent = formatFilterValue(clippedRange?.max);
  };

  let commitTimer = null;
  const commitNow = () => {
    if (commitTimer !== null) {
      window.clearTimeout(commitTimer);
      commitTimer = null;
    }
    onChanged();
  };
  const scheduleCommit = () => {
    if (commitTimer !== null) window.clearTimeout(commitTimer);
    // Coalesce the dense stream of range-input events. This still previews a
    // paused drag, while ensuring the latest value is applied even in notebook
    // iframes that do not reliably emit a final `change` event.
    commitTimer = window.setTimeout(() => {
      commitTimer = null;
      onChanged();
    }, 160);
  };
  const updateMin = (commit) => {
    setClipBoundary(clipState, 'min', sliderToFilterFrac(minInput.value, 0.0));
    sync();
    if (commit) commitNow();
    else scheduleCommit();
  };
  const updateMax = (commit) => {
    setClipBoundary(clipState, 'max', sliderToFilterFrac(maxInput.value, 1.0));
    sync();
    if (commit) commitNow();
    else scheduleCommit();
  };
  // Keep handles and labels responsive, coalesce live previews, and force the
  // latest value on every common interaction-completion event.
  minInput.addEventListener('input', () => updateMin(false));
  maxInput.addEventListener('input', () => updateMax(false));
  minInput.addEventListener('change', () => updateMin(true));
  maxInput.addEventListener('change', () => updateMax(true));
  minInput.addEventListener('pointerup', commitNow);
  maxInput.addEventListener('pointerup', commitNow);
  minInput.addEventListener('blur', commitNow);
  maxInput.addEventListener('blur', commitNow);

  sync();
  wrap.appendChild(title);
  wrap.appendChild(valuesLabel);
  wrap.appendChild(slider);
  return wrap;
}

function pickDefaultRuleProperty(scopeKeys = null) {
  const resolvedScopeKeys = resolveScopeKeys(scopeKeys, true);
  const options = getFilterPropertyOptions();
  for (const option of options) {
    const domain = getRuleDomain({ property: option.value }, resolvedScopeKeys);
    if (domain) return option.value;
  }
  return FILTER_POSITION;
}

function onFilterRulesUpdated() {
  updateFilterPanelUI();
  const needsFilterUpdate = layers.some((entry) => {
    if (entry.filterActive) return true;
    const activity = getEntryFilterActivity(entry);
    return activity.point || activity.edge || activity.face;
  });
  if (needsFilterUpdate) requestApplyFiltersToAllEntries();
}

function updateFilterPanelUI() {
  if (!filterRulesList) return;

  const scopeOptions = [];
  layers.forEach((entry, idx) => {
    scopeOptions.push({ value: getEntryScopeKey(entry), label: `${idx + 1}. ${entry.layer.label}` });
  });

  filterRulesList.innerHTML = '';
  if (!filterState.rules || filterState.rules.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'filter-empty-rules';
    empty.textContent = 'No active rules. Add one to start filtering.';
    filterRulesList.appendChild(empty);
  } else {
    const propertyOptions = getFilterPropertyOptions();
    filterState.rules.forEach((rule) => {
      if (Array.isArray(rule.scopeKeys)) {
        rule.scopeKeys = normalizeRuleScopeSelection(rule.scopeKeys, false);
      } else {
        rule.scopeKeys = normalizeRuleScopeSelection(rule.scopeKeys, true);
      }
      if (typeof rule.expanded !== 'boolean') rule.expanded = true;
      const row = document.createElement('div');
      row.className = 'filter-rule-row';
      const top = document.createElement('div');
      top.className = 'filter-rule-top';

      const body = document.createElement('div');
      body.className = 'filter-rule-body';
      body.classList.toggle('hidden', !rule.expanded);

      const expandButton = document.createElement('button');
      expandButton.type = 'button';
      expandButton.className = 'layer-options-toggle filter-rule-toggle';
      expandButton.setAttribute('aria-label', 'Toggle rule controls');
      expandButton.setAttribute('aria-expanded', rule.expanded ? 'true' : 'false');
      expandButton.textContent = rule.expanded ? '▾' : '▸';
      expandButton.addEventListener('click', () => {
        rule.expanded = !rule.expanded;
        expandButton.setAttribute('aria-expanded', rule.expanded ? 'true' : 'false');
        expandButton.textContent = rule.expanded ? '▾' : '▸';
        body.classList.toggle('hidden', !rule.expanded);
        if (!rule.expanded) closeAllDropdownMenus();
      });

      const propertyDropdown = createFilterDropdownControl({
        prefixText: 'Property:',
        options: propertyOptions,
        selectedValue: rule.property,
        onSelect: (nextProperty) => {
          if (rule.property === nextProperty) return;
          rule.property = nextProperty;
          rule.scalarClip = createClipState();
          rule.positionClip = {
            x: createClipState(),
            y: createClipState(),
            z: createClipState(),
          };
          onFilterRulesUpdated();
        },
        hostClassName: 'filter-rule-select-host',
      });

      const removeButton = document.createElement('button');
      removeButton.type = 'button';
      removeButton.className = 'filter-remove-rule-btn';
      removeButton.title = 'Remove rule';
      removeButton.textContent = '×';
      removeButton.addEventListener('click', () => {
        filterState.rules = filterState.rules.filter((candidate) => candidate.id !== rule.id);
        onFilterRulesUpdated();
      });

      const resetButton = document.createElement('button');
      resetButton.type = 'button';
      resetButton.className = 'filter-reset-rule-btn';
      resetButton.title = 'Reset rule';
      resetButton.textContent = '↺';
      resetButton.addEventListener('click', () => {
        resetRuleConfig(rule);
        onFilterRulesUpdated();
      });

      top.appendChild(expandButton);
      top.appendChild(propertyDropdown);
      top.appendChild(resetButton);
      top.appendChild(removeButton);
      row.appendChild(top);

      body.appendChild(createFilterMultiSelectDropdownControl({
        prefixText: 'Scope:',
        options: scopeOptions,
        selectedValues: rule.scopeKeys || [],
        onChange: (nextValues) => {
          rule.scopeKeys = nextValues;
          onFilterRulesUpdated();
        },
        hostClassName: 'filter-rule-scope-host',
      }));

      const domain = getRuleDomain(rule, rule.scopeKeys || []);
      if (!domain) {
        row.classList.add('disabled');
        const msg = document.createElement('div');
        msg.className = 'filter-empty-rules';
        msg.textContent = 'No matching values in current scope.';
        body.appendChild(msg);
      } else if (domain.kind === FILTER_POSITION) {
        const axisWrap = document.createElement('div');
        axisWrap.className = 'filter-position-grid';
        ['x', 'y', 'z'].forEach((axis) => {
          const sliderControl = createDualSliderControl({
            clipState: rule.positionClip[axis],
            range: domain.ranges[axis],
            label: `Position ${axis.toUpperCase()}`,
            onChanged: () => {
              requestApplyFiltersToAllEntries();
            },
          });
          const axisRow = document.createElement('div');
          axisRow.className = 'filter-position-axis';
          axisRow.appendChild(sliderControl);
          axisWrap.appendChild(axisRow);
        });
        body.appendChild(axisWrap);
      } else {
        body.appendChild(createDualSliderControl({
          clipState: rule.scalarClip,
          range: domain.range,
          label: domain.fieldName,
          onChanged: () => {
            requestApplyFiltersToAllEntries();
          },
        }));
      }
      row.appendChild(body);
      filterRulesList.appendChild(row);
    });
  }

  if (filterMaskedOpacityInput && filterMaskedOpacityValue) {
    filterMaskedOpacityInput.value = String(filterState.maskedOpacity);
    filterMaskedOpacityValue.textContent = Number(filterState.maskedOpacity).toFixed(2);
  }
}

function setFilterPanelOpen(nextOpen) {
  filterState.open = Boolean(nextOpen);
  if (filterPanel) {
    filterPanel.classList.toggle('collapsed', !filterState.open);
    filterPanel.setAttribute('aria-hidden', filterState.open ? 'false' : 'true');
  }
  if (filterToggleButton) {
    filterToggleButton.classList.toggle('active', filterState.open);
    filterToggleButton.textContent = filterState.open ? 'Filters On' : 'Filters';
    filterToggleButton.title = filterState.open ? 'Hide filters' : 'Show filters';
    filterToggleButton.setAttribute('aria-label', filterToggleButton.title);
  }
}

function initFilterPanelEvents() {
  if (filterToggleButton) {
    filterToggleButton.addEventListener('click', () => {
      setFilterPanelOpen(!filterState.open);
    });
  }
  if (filterCloseButton) {
    filterCloseButton.addEventListener('click', () => {
      setFilterPanelOpen(false);
    });
  }
  if (filterPanel) {
    filterPanel.addEventListener('scroll', () => {
      closeAllDropdownMenus();
    }, { passive: true });
  }
  if (filterAddRuleButton) {
    filterAddRuleButton.addEventListener('click', () => {
      const property = pickDefaultRuleProperty(getAllScopeKeys());
      filterState.rules.push(createDefaultRule(property));
      onFilterRulesUpdated();
    });
  }
  if (filterResetRulesButton) {
    filterResetRulesButton.addEventListener('click', () => {
      filterState.rules.forEach((rule) => {
        resetRuleConfig(rule);
      });
      onFilterRulesUpdated();
    });
  }
  if (filterMaskedOpacityInput) {
    filterMaskedOpacityInput.addEventListener('input', () => {
      const previousMaskedOpacity = filterState.maskedOpacity;
      filterState.maskedOpacity = clampUnit(filterMaskedOpacityInput.value, 0.0);
      if (filterMaskedOpacityValue) {
        filterMaskedOpacityValue.textContent = Number(filterState.maskedOpacity).toFixed(2);
      }
      for (const entry of layers) {
        updateEntryOpacity(entry, entry.opacity);
      }
      // Rejected mesh buffers are intentionally omitted while masked opacity
      // is zero. Materialize them once if the user asks to see masked faces.
      if (previousMaskedOpacity <= 0.0 && filterState.maskedOpacity > 0.0) {
        requestApplyFiltersToAllEntries();
      }
      requestRender();
    });
  }
  updateFilterPanelUI();
  setFilterPanelOpen(false);
}

function fitCamera() {
  const box = new THREE.Box3();
  let hasBox = false;
  for (const entry of layers) {
    if (!entry.visible) continue;
    if (entry.mesh?.visible) {
      box.expandByObject(entry.mesh);
      hasBox = true;
    }
    if (entry.maskedMesh?.visible) {
      box.expandByObject(entry.maskedMesh);
      hasBox = true;
    }
    if (entry.points?.visible) {
      box.expandByObject(entry.points);
      hasBox = true;
    }
    if (entry.maskedPoints?.visible) {
      box.expandByObject(entry.maskedPoints);
      hasBox = true;
    }
  }
  if (!hasBox || box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1.0;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const distance = (radius / Math.sin(fov / 2)) * 1.35;
  camera.position.set(center.x + distance, center.y + distance, center.z + distance);

  const near = Math.max(distance / 5000, 1e-6);
  const far = Math.max(distance * 50, 10.0);
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  surfaceOrbitPivot = center.clone();
  surfaceOrbitDragging = false;
  controls.update();
  requestRender();
}

function createCmapDropdown(entry, fieldIdx, controlsEnabled, association = 'vertex') {
  const dropdownWrap = document.createElement('div');
  dropdownWrap.className = 'layer-select-wrapper layer-field-controls';
  const controls = document.createElement('div');
  controls.className = 'field-select-inline';
  const prevButton = document.createElement('button');
  prevButton.type = 'button';
  prevButton.className = 'field-nav-button';
  prevButton.textContent = '←';
  prevButton.setAttribute('aria-label', 'Previous colormap');
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-dropdown';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'layer-select-button cmap-dropdown-button';
  button.disabled = !controlsEnabled;
  const options = getEntryColorModeOptions(entry, fieldIdx, association);
  const currentCmap = options.includes(getEntryCmap(entry, fieldIdx, association)) ? getEntryCmap(entry, fieldIdx, association) : UNIFORM_CMAP;
  setCmapButtonContent(button, currentCmap);
  const menu = document.createElement('div');
  menu.className = 'custom-dropdown-menu';
  const nextButton = document.createElement('button');
  nextButton.type = 'button';
  nextButton.className = 'field-nav-button';
  nextButton.textContent = '→';
  nextButton.setAttribute('aria-label', 'Next colormap');

  const updateCmapControls = (cmapName) => {
    const cmapIdx = Math.max(0, options.indexOf(cmapName));
    setCmapButtonContent(button, options[cmapIdx]);
    menu.querySelectorAll('.custom-dropdown-item').forEach((el, idx) => {
      el.classList.toggle('selected', idx === cmapIdx);
    });
    button.disabled = !controlsEnabled;
    if (!controlsEnabled) closeDropdownMenu(menu);
    prevButton.disabled = !controlsEnabled || cmapIdx <= 0;
    nextButton.disabled = !controlsEnabled || cmapIdx >= options.length - 1;
  };

  const selectCmap = (cmapName) => {
    if (!controlsEnabled) return;
    const nextCmap = options.includes(cmapName) ? cmapName : UNIFORM_CMAP;
    setLayerCmap(entry, nextCmap, association);
    updateCmapControls(nextCmap);
  };

  const moveCmap = (delta) => {
    if (!controlsEnabled) return;
    const currentIdx = Math.max(0, options.indexOf(getEntryCmap(entry, fieldIdx, association)));
    const nextIdx = Math.max(0, Math.min(options.length - 1, currentIdx + delta));
    if (nextIdx === currentIdx) return;
    closeDropdownMenu(menu);
    selectCmap(options[nextIdx]);
  };

  options.forEach((cmapName) => {
    const item = document.createElement('div');
    item.className = 'custom-dropdown-item cmap-dropdown-item';
    if (!controlsEnabled) item.classList.add('disabled');
    if (cmapName === currentCmap) item.classList.add('selected');
    const prefix = document.createElement('span');
    prefix.className = 'cmap-dropdown-prefix';
    prefix.appendChild(createCmapSwatch(entry, fieldIdx, cmapName, association));
    item.appendChild(prefix);
    const value = document.createElement('span');
    value.className = 'layer-id';
    value.textContent = cmapName || UNIFORM_CMAP;
    item.appendChild(value);
    item.addEventListener('click', () => {
      if (!controlsEnabled) return;
      selectCmap(cmapName);
      closeDropdownMenu(menu);
    });
    menu.appendChild(item);
  });
  prevButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    moveCmap(-1);
  });
  button.addEventListener('click', (ev) => {
    if (!controlsEnabled) return;
    ev.stopPropagation();
    toggleDropdownMenu(button, menu);
  });
  nextButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    moveCmap(1);
  });
  updateCmapControls(currentCmap);
  controls.appendChild(prevButton);
  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  controls.appendChild(dropdown);
  controls.appendChild(nextButton);
  dropdownWrap.appendChild(controls);
  return { dropdownWrap, currentCmap };
}

function getAssociationLabel(entry, association) {
  if (association === 'edge') return 'Edge';
  if (association === 'face') return meshFaceSize(entry.layer) === 3 ? 'Triangle' : 'Face';
  return entry.layerKind === 'pointcloud' ? 'Point' : 'Vertex';
}

function createEntryFieldPanel(entry, association = 'vertex') {
  const state = getAssociationState(entry, association);
  const fieldIdx = state.activeFieldIdx ?? 0;
  const fieldNames = getAssociationFieldData(entry, association).names;
  const meta = getScalarMeta(entry, fieldIdx, association);
  const rawColors = getEntryRawColors(entry, fieldIdx, association);
  const hasDirectColors = rawColors && typeof rawColors.length === 'number' && rawColors.length > 0;
  if (!meta && !entry.layer?.texture && !hasDirectColors) return null;

  const fieldLabel = fieldNames ? fieldNames[fieldIdx] : null;
  const associationLabel = getAssociationLabel(entry, association);
  const colorbarLabel = meta ? (fieldLabel ? `${associationLabel}: ${fieldLabel}` : `${associationLabel} colors`) : `${associationLabel} color mode`;
  const controlsEnabled = entry.visible;

  const wrapper = document.createElement('div');
  wrapper.className = 'colorbar';
  const title = document.createElement('div');
  title.className = 'colorbar-title';
  title.textContent = colorbarLabel;
  const { dropdownWrap, currentCmap } = createCmapDropdown(entry, fieldIdx, controlsEnabled, association);
  wrapper.appendChild(title);
  wrapper.appendChild(dropdownWrap);
  if (!meta) {
    if (hasDirectColors) {
      const visual = document.createElement('div');
      visual.className = 'colorbar-visual';
      const gradientWrap = document.createElement('div');
      gradientWrap.className = 'colorbar-gradient-wrap';
      const gradient = document.createElement('div');
      gradient.className = 'colorbar-gradient';
      gradient.style.background = gradientCss(getEntryRawGradient(entry, fieldIdx, association));
      gradientWrap.appendChild(gradient);
      visual.appendChild(gradientWrap);
      wrapper.appendChild(visual);
    }
    return wrapper;
  }
  const visual = document.createElement('div');
  visual.className = 'colorbar-visual';
  const distributionWrap = document.createElement('div');
  distributionWrap.className = 'colorbar-distribution-wrap';
  const distributionViz = buildDistributionSvg(
    getEntryDisplayedValues(entry, fieldIdx, association),
    { min: meta.fullMin, max: meta.fullMax },
  );
  const distribution = distributionViz ? distributionViz.svg : null;
  const gradientWrap = document.createElement('div');
  gradientWrap.className = 'colorbar-gradient-wrap';
  const gradient = document.createElement('div');
  gradient.className = 'colorbar-gradient';
  gradient.style.background = gradientCss(meta.gradient);
  if (distribution) {
    distributionWrap.appendChild(distribution);
  }
  visual.appendChild(distributionWrap);
  gradientWrap.appendChild(gradient);
  wrapper.appendChild(visual);
  if (Number.isFinite(meta.min) && Number.isFinite(meta.max)) {
    const range = document.createElement('div');
    range.className = 'colorbar-range';
    const minLabel = document.createElement('span');
    const maxLabel = document.createElement('span');
    range.appendChild(minLabel);
    range.appendChild(maxLabel);
    wrapper.appendChild(range);

    if (usesScalarCmap(currentCmap)) {
      const clipWrap = document.createElement('div');
      clipWrap.className = 'colorbar-clip';
      const clipSlider = document.createElement('div');
      clipSlider.className = 'colorbar-clip-slider';
      const clipTrack = document.createElement('div');
      clipTrack.className = 'colorbar-clip-track';
      const clipFill = document.createElement('div');
      clipFill.className = 'colorbar-clip-fill';
      const minInput = document.createElement('input');
      minInput.type = 'range';
      minInput.min = '0';
      minInput.max = String(CLIP_SLIDER_MAX);
      minInput.step = '1';
      minInput.setAttribute('aria-label', `Minimum clip for ${colorbarLabel}`);
      const maxInput = document.createElement('input');
      maxInput.type = 'range';
      maxInput.min = '0';
      maxInput.max = String(CLIP_SLIDER_MAX);
      maxInput.step = '1';
      maxInput.setAttribute('aria-label', `Maximum clip for ${colorbarLabel}`);

      clipSlider.appendChild(clipTrack);
      clipSlider.appendChild(clipFill);
      clipSlider.appendChild(minInput);
      clipSlider.appendChild(maxInput);
      clipWrap.appendChild(clipSlider);
      visual.appendChild(clipWrap);

      const setActiveClipHandle = (handleName) => {
        const activeHandle = handleName === 'min' ? 'min' : 'max';
        clipSlider.dataset.activeHandle = activeHandle;
        minInput.style.zIndex = activeHandle === 'min' ? '3' : '2';
        maxInput.style.zIndex = activeHandle === 'max' ? '3' : '2';
      };

      const updateClipHandlePriority = (clientX = null) => {
        const rect = clipSlider.getBoundingClientRect();
        if (!(rect.width > 0)) {
          setActiveClipHandle(clipSlider.dataset.activeHandle || 'max');
          return;
        }
        const currentMeta = getScalarMeta(entry, fieldIdx, association);
        const clipState = currentMeta?.clipState || createClipState();
        const minPos = rect.width * clampUnit(clipState.minFrac, 0.0);
        const maxPos = rect.width * clampUnit(clipState.maxFrac, 1.0);
        let activeHandle = clipSlider.dataset.activeHandle || 'max';

        if (Number.isFinite(clientX)) {
          const x = Math.max(0.0, Math.min(rect.width, clientX - rect.left));
          if (Math.abs(maxPos - minPos) <= 14) {
            activeHandle = x <= minPos ? 'min' : 'max';
          } else {
            activeHandle = Math.abs(x - minPos) <= Math.abs(x - maxPos) ? 'min' : 'max';
          }
        }

        setActiveClipHandle(activeHandle);
      };

      clipSlider.addEventListener('pointerdown', (ev) => {
        updateClipHandlePriority(ev.clientX);
      });
      clipSlider.addEventListener('pointermove', (ev) => {
        if ((ev.buttons || 0) === 0) {
          updateClipHandlePriority(ev.clientX);
        }
      });
      minInput.addEventListener('pointerdown', () => {
        setActiveClipHandle('min');
      });
      maxInput.addEventListener('pointerdown', () => {
        setActiveClipHandle('max');
      });

      const syncClipControl = () => {
        const currentMeta = getScalarMeta(entry, fieldIdx, association);
        const clipState = currentMeta?.clipState || createClipState();
        const startPct = clampUnit(clipState.minFrac, 0.0) * 100.0;
        const endPct = clampUnit(clipState.maxFrac, 1.0) * 100.0;
        clipFill.style.left = `${startPct}%`;
        clipFill.style.width = `${Math.max(0.0, endPct - startPct)}%`;
        distributionViz?.setClipFractions(clipState.minFrac, clipState.maxFrac);
        minInput.value = String(clipToSliderValue(clipState.minFrac, 0.0));
        maxInput.value = String(clipToSliderValue(clipState.maxFrac, 1.0));
        minInput.disabled = !controlsEnabled || !currentMeta?.clipEnabled;
        maxInput.disabled = !controlsEnabled || !currentMeta?.clipEnabled;
        minLabel.textContent = formatScalarValue(currentMeta?.min);
        maxLabel.textContent = formatScalarValue(currentMeta?.max);
        updateClipHandlePriority();
      };

      minInput.addEventListener('input', () => {
        setColorClipBoundary(entry, fieldIdx, 'min', minInput.value, association);
      });
      maxInput.addEventListener('input', () => {
        setColorClipBoundary(entry, fieldIdx, 'max', maxInput.value, association);
      });

      wrapper.__syncClipControl = syncClipControl;
      syncClipControl();
    } else {
      minLabel.textContent = formatScalarValue(meta.min);
      maxLabel.textContent = formatScalarValue(meta.max);
      distributionViz?.setClipFractions(0.0, 1.0);
    }
    visual.appendChild(gradientWrap);
  }

  return wrapper;
}

function updateColorbars() {
  for (const entry of layers) {
    const containers = entry.fieldOptionsContainers || {};
    Object.entries(containers).forEach(([association, container]) => {
      container.innerHTML = '';
      const panel = createEntryFieldPanel(entry, association);
      if (panel) container.appendChild(panel);
    });
  }
}

function updateInfoPanel() {
  let totalPoints = 0;
  let totalVertices = 0;
  let totalFaces = 0;
  for (const entry of layers) {
    if (!entry.visible) continue;
    if (entry.layerKind === 'pointcloud') {
      totalPoints += Number(entry.layer.vertex_count) || 0;
    } else {
      totalVertices += Number(entry.layer.vertex_count) || 0;
    }
    totalFaces += Number(entry.layer.face_count) || 0;
  }
  const pointsNode = document.getElementById('info-points');
  const verticesNode = document.getElementById('info-vertices');
  const facesNode = document.getElementById('info-faces');
  if (pointsNode) pointsNode.textContent = totalPoints.toLocaleString();
  if (verticesNode) verticesNode.textContent = totalVertices.toLocaleString();
  if (facesNode) facesNode.textContent = totalFaces.toLocaleString();
}

function setLayerField(entry, fieldIdx, association = 'vertex') {
  const fieldNames = getAssociationFieldData(entry, association).names;
  if (!fieldNames || fieldIdx < 0 || fieldIdx >= fieldNames.length) return;
  const previousSurfaceAssociation = entry.activeSurfaceAssociation;
  renderEntryColors(entry, fieldIdx, association);
  if (
    entry.layerKind === 'mesh'
    && association !== 'edge'
    && previousSurfaceAssociation !== entry.activeSurfaceAssociation
  ) {
    rebuildLayerList();
  } else {
    updateColorbars();
  }
}


function createRangeOptionControl({
  labelText,
  min,
  max,
  step,
  value,
  disabled,
  onInput,
}) {
  const wrap = document.createElement('div');
  wrap.className = 'layer-option-control';
  const label = document.createElement('label');
  label.textContent = labelText;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.disabled = disabled;
  input.addEventListener('input', () => onInput(input.value));
  wrap.appendChild(label);
  wrap.appendChild(input);
  return wrap;
}

function appendPointLayerOptions(entry, optionGrid) {
  optionGrid.appendChild(createRangeOptionControl({
    labelText: 'Point size',
    min: 1,
    max: 12,
    step: 0.5,
    value: entry.pointSize,
    disabled: !entry.visible,
    onInput: (nextValue) => {
      updateEntryPointSize(entry, nextValue);
      requestRender();
    },
  }));

  optionGrid.appendChild(createRangeOptionControl({
    labelText: 'Point opacity',
    min: 0,
    max: 1,
    step: 0.05,
    value: entry.opacity,
    disabled: !entry.visible,
    onInput: (nextValue) => {
      updateEntryOpacity(entry, nextValue);
      requestRender();
    },
  }));

  if (entry.pointLines && entry.visible) {
    optionGrid.appendChild(createRangeOptionControl({
      labelText: 'Edge thickness',
      min: 1,
      max: 12,
      step: 0.5,
      value: entry.edgeThickness,
      disabled: !entry.visible,
      onInput: (nextValue) => {
        updateEntryEdgeThickness(entry, nextValue);
        requestRender();
      },
    }));
  }

}

function appendMeshLayerOptions(entry, optionGrid) {
  optionGrid.appendChild(createRangeOptionControl({
    labelText: 'Mesh opacity',
    min: 0,
    max: 1,
    step: 0.05,
    value: entry.opacity,
    disabled: !entry.visible,
    onInput: (nextValue) => {
      updateEntryOpacity(entry, nextValue);
      requestRender();
    },
  }));

  const toggleRow = document.createElement('div');
  toggleRow.className = 'layer-option-toggle-row';

  const edgeToggleLabel = document.createElement('label');
  edgeToggleLabel.className = 'layer-option-toggle';
  const edgeToggleInput = document.createElement('input');
  edgeToggleInput.type = 'checkbox';
  edgeToggleInput.checked = shouldDisplayMeshEdges(entry);
  edgeToggleInput.disabled = !entry.visible || isMeshEdgeFieldActive(entry);
  edgeToggleInput.addEventListener('change', () => {
    if (isMeshEdgeFieldActive(entry)) return;
    entry.showEdges = edgeToggleInput.checked;
    if (entry.showEdges && entry.filterActive && entry.layer.edges?.length) {
      rebuildMeshEdgeOverlay(entry);
    } else {
      updateEntryEdgeOverlay(entry);
    }
    rebuildLayerList();
    requestRender();
  });
  edgeToggleLabel.appendChild(edgeToggleInput);
  edgeToggleLabel.appendChild(document.createTextNode('Show edges'));
  toggleRow.appendChild(edgeToggleLabel);

  const vertexToggleLabel = document.createElement('label');
  vertexToggleLabel.className = 'layer-option-toggle';
  const vertexToggleInput = document.createElement('input');
  vertexToggleInput.type = 'checkbox';
  vertexToggleInput.checked = Boolean(entry.showVertices);
  vertexToggleInput.disabled = !entry.visible;
  vertexToggleInput.addEventListener('change', () => {
    entry.showVertices = vertexToggleInput.checked;
    if (entry.showVertices) {
      ensureMeshVertexPoints(entry);
    }
    updateEntryVertexOverlay(entry);
    rebuildLayerList();
    requestRender();
  });
  vertexToggleLabel.appendChild(vertexToggleInput);
  vertexToggleLabel.appendChild(document.createTextNode('Show vertices'));
  toggleRow.appendChild(vertexToggleLabel);
  optionGrid.appendChild(toggleRow);

  if (
    entry.edgeLines
    && shouldDisplayMeshEdges(entry)
    && entry.visible
  ) {
    optionGrid.appendChild(createRangeOptionControl({
      labelText: 'Edge opacity',
      min: 0,
      max: 1,
      step: 0.05,
      value: entry.edgeOpacity ?? (isDayMode ? 0.82 : 0.92),
      disabled: !entry.visible,
      onInput: (nextValue) => {
        entry.edgeOpacity = clampOpacity(nextValue);
        updateEntryEdgeOverlay(entry);
        requestRender();
      },
    }));
    optionGrid.appendChild(createRangeOptionControl({
      labelText: 'Edge thickness',
      min: 1,
      max: 12,
      step: 0.5,
      value: entry.edgeThickness,
      disabled: !entry.visible,
      onInput: (nextValue) => {
        updateEntryEdgeThickness(entry, nextValue);
        requestRender();
      },
    }));
  }

  if (entry.showVertices && entry.visible) {
    optionGrid.appendChild(createRangeOptionControl({
      labelText: 'Point size',
      min: 1,
      max: 12,
      step: 0.5,
      value: entry.pointSize,
      disabled: !entry.visible,
      onInput: (nextValue) => {
        ensureMeshVertexPoints(entry);
        updateEntryPointSize(entry, nextValue);
        requestRender();
      },
    }));
  }
}

function appendShareScaleOption(entry, optionGrid) {
  const fieldCount = ['vertex', 'edge', 'face']
    .reduce(
      (total, association) => total + (hasEntryFields(entry, association) ? getEntryFieldCount(entry, association) : 0),
      0,
    );
  if (fieldCount <= 1) return;
  const shareScaleLabel = document.createElement('label');
  shareScaleLabel.className = 'layer-option-toggle';
  const shareScaleInput = document.createElement('input');
  shareScaleInput.type = 'checkbox';
  shareScaleInput.checked = Boolean(entry.shareFieldScale);
  shareScaleInput.disabled = !entry.visible;
  shareScaleInput.addEventListener('change', () => {
    setLayerShareFieldScale(entry, shareScaleInput.checked);
  });
  shareScaleLabel.appendChild(shareScaleInput);
  shareScaleLabel.appendChild(document.createTextNode('Share scale within each element type'));
  optionGrid.appendChild(shareScaleLabel);
}

function createLayerFieldSelector(entry, association = 'vertex') {
  const fieldNames = getAssociationFieldData(entry, association).names;
  if (fieldNames.length <= 1) return null;
  const state = getAssociationState(entry, association);

  const wrap = document.createElement('div');
  wrap.className = 'layer-select-wrapper layer-field-controls';
  const controls = document.createElement('div');
  controls.className = 'field-select-inline';
  const prevButton = document.createElement('button');
  prevButton.type = 'button';
  prevButton.className = 'field-nav-button';
  prevButton.textContent = '←';
  prevButton.setAttribute('aria-label', 'Previous field');
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-dropdown';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'layer-select-button';
  const menu = document.createElement('div');
  menu.className = 'custom-dropdown-menu';
  const nextButton = document.createElement('button');
  nextButton.type = 'button';
  nextButton.className = 'field-nav-button';
  nextButton.textContent = '→';
  nextButton.setAttribute('aria-label', 'Next field');

  const updateFieldControls = (fieldIdx) => {
    setFieldButtonContent(button, fieldNames[fieldIdx]);
    menu.querySelectorAll('.custom-dropdown-item').forEach((el, idx2) => {
      el.classList.toggle('selected', idx2 === fieldIdx);
    });
    button.disabled = !entry.visible;
    if (!entry.visible) closeDropdownMenu(menu);
    prevButton.disabled = !entry.visible || fieldIdx <= 0;
    nextButton.disabled = !entry.visible || fieldIdx >= fieldNames.length - 1;
  };

  const selectField = (fieldIdx) => {
    const nextFieldIdx = Math.max(0, Math.min(fieldNames.length - 1, fieldIdx));
    setLayerField(entry, nextFieldIdx, association);
    updateFieldControls(nextFieldIdx);
  };

  const moveField = (delta) => {
    if (!entry.visible) return false;
    const currentIdx = state.activeFieldIdx ?? 0;
    const nextIdx = Math.max(0, Math.min(fieldNames.length - 1, currentIdx + delta));
    if (nextIdx === currentIdx) return false;
    closeDropdownMenu(menu);
    selectField(nextIdx);
    return true;
  };

  let repeatDelay = 0;
  let repeatInterval = 0;
  let suppressNextClick = false;
  const stopFieldRepeat = () => {
    clearTimeout(repeatDelay);
    clearInterval(repeatInterval);
    repeatDelay = 0;
    repeatInterval = 0;
  };
  const startFieldRepeat = (buttonEl, delta, ev) => {
    if (buttonEl.disabled || ev.button !== 0) return;
    stopFieldRepeat();
    buttonEl.setPointerCapture?.(ev.pointerId);
    repeatDelay = window.setTimeout(() => {
      suppressNextClick = true;
      if (!moveField(delta)) return;
      repeatInterval = window.setInterval(() => {
        if (!moveField(delta)) stopFieldRepeat();
      }, 120);
    }, 320);
  };

  fieldNames.forEach((name, fieldIdx) => {
    const item = document.createElement('div');
    item.className = 'custom-dropdown-item';
    if (fieldIdx === (state.activeFieldIdx ?? 0)) item.classList.add('selected');
    const value = document.createElement('span');
    value.className = 'layer-id';
    value.textContent = name;
    item.appendChild(value);
    item.addEventListener('click', () => {
      selectField(fieldIdx);
      closeDropdownMenu(menu);
    });
    menu.appendChild(item);
  });

  prevButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    moveField(-1);
  });
  prevButton.addEventListener('pointerdown', (ev) => startFieldRepeat(prevButton, -1, ev));
  button.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleDropdownMenu(button, menu);
  });
  nextButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    moveField(1);
  });
  nextButton.addEventListener('pointerdown', (ev) => startFieldRepeat(nextButton, 1, ev));
  [prevButton, nextButton].forEach((navButton) => {
    navButton.addEventListener('pointerup', stopFieldRepeat);
    navButton.addEventListener('pointercancel', stopFieldRepeat);
    navButton.addEventListener('lostpointercapture', stopFieldRepeat);
  });

  updateFieldControls(Math.max(0, Math.min(fieldNames.length - 1, state.activeFieldIdx ?? 0)));
  controls.appendChild(prevButton);
  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  controls.appendChild(dropdown);
  controls.appendChild(nextButton);
  wrap.appendChild(controls);
  return wrap;
}

function rebuildLayerList() {
  const container = document.getElementById('layer-list');
  container.innerHTML = '';
  layers.forEach((entry, idx) => {
    const row = document.createElement('div');
    row.className = 'layer-row';
    const isPointLayer = entry.layerKind === 'pointcloud';
    const hasFieldOptions = true;
    const hasScalarOptions = getEntryFieldCount(entry) > 0
      || Boolean(entry.layer.scalar_meta)
      || Boolean(entry.layer.values)
      || Boolean(entry.layer.texture);
    let controlsBody = null;
    let toggleButton = null;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = entry.visible;
    checkbox.disabled = entry.loading;
    checkbox.addEventListener('change', async () => {
      entry.visible = checkbox.checked;
      if (entry.visible && !entry.loaded) {
        try {
          await loadLayerEntry(entry);
        } catch (error) {
          console.error(`Failed to load layer ${entry.layer.label}`, error);
        }
      }
      if (entry.mesh) entry.mesh.visible = entry.visible;
      if (entry.interactionProxy) entry.interactionProxy.visible = false;
      if (entry.points) entry.points.visible = entry.visible;
      updateMaskedObjectVisibility(entry);
      updateEntryEdgeOverlay(entry);
      rebuildLayerList();
      requestRender();
    });

    const info = document.createElement('div');
    info.className = 'layer-main';
    const label = document.createElement('div');
    label.className = 'layer-label';
    const loadState = entry.loading
      ? ' (loading…)'
      : (entry.loadError ? ' (load failed)' : '');
    label.textContent = `${idx + 1}. ${entry.layer.label}${loadState}`;
    info.appendChild(label);

    if (hasFieldOptions) {
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'layer-options-toggle';
      toggle.setAttribute('aria-label', `Toggle field options for ${entry.layer.label}`);
      toggle.setAttribute('aria-expanded', entry.optionsExpanded ? 'true' : 'false');
      toggle.textContent = entry.optionsExpanded ? '▾' : '▸';
      toggle.addEventListener('click', () => {
        entry.optionsExpanded = !entry.optionsExpanded;
        toggle.setAttribute('aria-expanded', entry.optionsExpanded ? 'true' : 'false');
        toggle.textContent = entry.optionsExpanded ? '▾' : '▸';
        controlsBody?.classList.toggle('hidden', !entry.optionsExpanded);
        if (!entry.optionsExpanded) closeAllDropdownMenus();
      });
      toggleButton = toggle;
    }

    controlsBody = document.createElement('div');
    controlsBody.className = 'layer-controls-body';
    controlsBody.classList.toggle('hidden', !entry.optionsExpanded);
    controlsBody.classList.toggle('disabled', !entry.visible);

    const optionGrid = document.createElement('div');
    optionGrid.className = 'layer-option-grid';

    if (isPointLayer) {
      appendPointLayerOptions(entry, optionGrid);
    } else {
      appendMeshLayerOptions(entry, optionGrid);
    }

    appendShareScaleOption(entry, optionGrid);

    controlsBody.appendChild(optionGrid);

    entry.fieldOptionsContainers = {};
    const associations = [];
    if (hasEntryFields(entry, 'vertex')) associations.push('vertex');
    if (hasEntryFields(entry, 'edge')) associations.push('edge');
    if (hasEntryFields(entry, 'face')) associations.push('face');
    associations.forEach((association) => {
      const hasAssociationFields = getEntryFieldCount(entry, association) > 0;
      if (!hasAssociationFields && association !== 'vertex') return;
      const section = document.createElement('div');
      section.className = `layer-field-section field-association-${association}`;
      const heading = document.createElement('div');
      heading.className = 'layer-field-association-title';
      if (getEntryFieldCount(entry, association) > 0) {
        const activeInput = document.createElement('input');
        activeInput.type = 'radio';
        activeInput.name = `active-viewer-field-${entry.layerId}`;
        activeInput.dataset.layerId = String(entry.layerId);
        activeInput.dataset.association = association;
        activeInput.dataset.fieldAssociation = 'true';
        activeInput.checked = entry.activeFieldAssociation === association;
        activeInput.disabled = !entry.visible;
        activeInput.addEventListener('click', (event) => {
          event.preventDefault();
          if (entry.activeFieldAssociation === association) {
            clearEntryFieldDisplay(entry);
          } else {
            const state = getAssociationState(entry, association);
            renderEntryColors(entry, state.activeFieldIdx ?? 0, association);
          }
          rebuildLayerList();
        });
        heading.appendChild(activeInput);
      }
      heading.appendChild(document.createTextNode(`${getAssociationLabel(entry, association)} fields`));
      section.appendChild(heading);
      const fieldSelector = createLayerFieldSelector(entry, association);
      if (fieldSelector) section.appendChild(fieldSelector);
      const colorbarHost = document.createElement('div');
      colorbarHost.className = 'layer-colorbar-host';
      section.appendChild(colorbarHost);
      entry.fieldOptionsContainers[association] = colorbarHost;
      controlsBody.appendChild(section);
    });

    row.appendChild(checkbox);
    row.appendChild(info);
    if (toggleButton) row.appendChild(toggleButton);
    if (hasFieldOptions) row.appendChild(controlsBody);
    container.appendChild(row);
  });

  updateColorbars();
  updateInfoPanel();
  updateToggleAllLayers();
  updateFilterPanelUI();
}

function updateToggleAllLayers() {
  const toggleAll = document.getElementById('toggle-all-layers');
  if (!toggleAll) return;
  if (layers.length === 0) {
    toggleAll.checked = false;
    toggleAll.indeterminate = false;
    toggleAll.disabled = true;
    return;
  }

  const visibleCount = layers.reduce((acc, entry) => acc + (entry.visible ? 1 : 0), 0);
  toggleAll.disabled = false;
  toggleAll.checked = visibleCount === layers.length;
  toggleAll.indeterminate = visibleCount > 0 && visibleCount < layers.length;
}

async function setAllLayersVisibility(visible) {
  for (const entry of layers) {
    entry.visible = visible;
    if (visible && !entry.loaded) {
      await loadLayerEntry(entry);
    }
    if (entry.mesh) entry.mesh.visible = visible;
    if (entry.interactionProxy) entry.interactionProxy.visible = false;
    if (entry.points) entry.points.visible = visible;
    updateMaskedObjectVisibility(entry);
    updateEntryEdgeOverlay(entry);
  }
  rebuildLayerList();
  requestRender();
}

document.addEventListener('click', (ev) => {
  if (!ev.target.closest('.custom-dropdown') && !ev.target.closest('.custom-dropdown-menu')) {
    closeAllDropdownMenus();
  }
});

function makeLayerPlaceholder(rawLayer, layerIdx) {
  const layerKind = String(rawLayer.layer_kind || (rawLayer.points ? 'pointcloud' : 'mesh')).toLowerCase() === 'pointcloud'
    ? 'pointcloud'
    : 'mesh';
  const layer = {
    ...rawLayer,
    vertices: new Float32Array(0),
    faces: new Uint32Array(0),
    points: new Float32Array(0),
    edges: null,
    colors: new Uint8Array(0),
    raw_colors: new Uint8Array(0),
    values: null,
    field_names: null,
    field_colors: null,
    field_values: null,
    field_scalar_meta: null,
    edge_field_names: null,
    edge_field_colors: null,
    edge_field_values: null,
    edge_field_scalar_meta: null,
    face_field_names: null,
    face_field_colors: null,
    face_field_values: null,
    face_field_scalar_meta: null,
    face_field_sorted_ascending: null,
    scalar_meta: null,
    texture: null,
  };
  return {
    layer,
    sourceLayer: rawLayer,
    layerId: layerIdx,
    layerKind,
    loaded: false,
    loading: false,
    loadError: null,
    mesh: null,
    maskedMesh: null,
    edgeLines: null,
    vertexPoints: null,
    maskedVertexPoints: null,
    interactionProxy: null,
    textureMap: null,
    points: null,
    maskedPoints: null,
    pointLines: null,
    visible: rawLayer.visible !== false,
    activeFieldIdx: Number.isInteger(rawLayer.active_field_idx) ? rawLayer.active_field_idx : 0,
    activeEdgeFieldIdx: Number.isInteger(rawLayer.active_edge_field_idx) ? rawLayer.active_edge_field_idx : 0,
    activeFaceFieldIdx: Number.isInteger(rawLayer.active_face_field_idx) ? rawLayer.active_face_field_idx : 0,
    activeValues: null,
    activeCmap: rawLayer.active_cmap || UNIFORM_CMAP,
    currentColors: new Uint8Array(0),
    currentEdgeColors: null,
    currentFaceColors: null,
    activeFieldAssociation: null,
    activeSurfaceAssociation: null,
    faceColorMaterialActive: false,
    unmaskedPointCount: 0,
    maskedPointCount: 0,
    unmaskedEdgeCount: 0,
    unmaskedFaceCount: 0,
    maskedFaceCount: 0,
    fieldClipStates: [createClipState()],
    sharedFieldClip: createClipState(),
    shareFieldScale: Boolean(rawLayer.share_field_scale ?? payload.initial_share_field_scale),
    opacity: clampOpacity(rawLayer.opacity ?? payload.opacity),
    edgeOpacity: rawLayer.edge_opacity == null ? null : clampOpacity(rawLayer.edge_opacity),
    showEdges: layerKind === 'mesh' && rawLayer.show_edges === true,
    showVertices: layerKind === 'mesh' && rawLayer.show_vertices === true,
    edgeThickness: getLayerEdgeThickness(rawLayer),
    pointSize: Math.min(12.0, Math.max(1.0, Number(rawLayer.point_size ?? payload.point_size) || 5.0)),
    optionsExpanded: rawLayer.visible !== false,
    filterActive: false,
    edgePassMask: null,
    facePassMask: null,
    pointFilterActive: false,
    edgeFilterActive: false,
    faceFilterActive: false,
    sortedFaceFilterActive: false,
    sortedFaceFilterUsesFaceColors: false,
    sortedFaceFilterBegin: null,
    sortedFaceFilterEnd: null,
    fieldOptionsContainer: null,
    fieldLookup: new Map(),
    edgeFieldLookup: new Map(),
    faceFieldLookup: new Map(),
    associationFieldStates: {},
    fieldOptionsContainers: {},
  };
}

async function loadLayerEntry(entry) {
  if (entry.loaded || entry.loading) return;
  entry.loading = true;
  entry.loadError = null;
  setLoadingStatus(`Loading ${entry.layer.label}…`);
  rebuildLayerList();
  try {
    await hydrateLayerArrays(entry.sourceLayer);
    const layer = normalizeLayerData(entry.sourceLayer);
    const layerKind = getLayerKind(layer);
    const built = layerKind === 'pointcloud' ? buildPoints(layer) : buildMesh(layer);
  const fieldMetas = layer.field_scalar_meta || null;
  const fieldValues = layer.field_values || null;
  const fieldColorCount = Array.isArray(layer.field_colors) ? layer.field_colors.length : 0;
  const fieldCount = Math.max(
    Array.isArray(layer.field_names) ? layer.field_names.length : 0,
    Array.isArray(fieldValues) ? fieldValues.length : 0,
    Array.isArray(fieldMetas) ? fieldMetas.length : 0,
    fieldColorCount,
  );
    Object.assign(entry, {
      layer,
      layerKind,
      loaded: true,
      mesh: layerKind === 'mesh' ? built.mesh : null,
      maskedMesh: layerKind === 'mesh' ? built.maskedMesh : null,
      edgeLines: layerKind === 'mesh' ? built.edgeLines : null,
      textureMap: layerKind === 'mesh' ? built.textureMap : null,
      points: layerKind === 'pointcloud' ? built.points : null,
      maskedPoints: layerKind === 'pointcloud' ? built.maskedPoints : null,
      pointLines: layerKind === 'pointcloud' ? built.lines : null,
      activeValues: layer.values || null,
      activeCmap: layer.active_cmap || (layer.scalar_meta && layer.scalar_meta.cmap) || UNIFORM_CMAP,
      currentColors: decodeArraySpec(layer.colors, Uint8Array) || new Uint8Array(0),
      currentEdgeColors: null,
      currentFaceColors: null,
      unmaskedPointCount: layerKind === 'pointcloud' ? Math.floor(layer.points.length / 3) : 0,
      unmaskedEdgeCount: pointGraphEdgeCount(layer),
      unmaskedFaceCount: layerKind === 'mesh' ? Math.floor(layer.faces.length / meshFaceSize(layer)) : 0,
      fieldClipStates: Array.from({ length: Math.max(fieldCount, 1) }, () => createClipState()),
    });
    entry.fieldLookup.clear();
  const fieldNamesForLookup = Array.isArray(layer.field_names) ? layer.field_names : [];
  if (Array.isArray(fieldValues)) {
    fieldNamesForLookup.forEach((name, idx) => {
      const values = fieldValues[idx];
      if (values) entry.fieldLookup.set(String(name), values);
    });
  }
  entry.edgeFieldLookup.clear();
  const edgeFieldNamesForLookup = Array.isArray(layer.edge_field_names) ? layer.edge_field_names : [];
  const edgeFieldValues = layer.edge_field_values || null;
  if (Array.isArray(edgeFieldValues)) {
    edgeFieldNamesForLookup.forEach((name, idx) => {
      const values = edgeFieldValues[idx];
      if (values) entry.edgeFieldLookup.set(String(name), values);
    });
  }
  entry.faceFieldLookup.clear();
  const faceFieldNamesForLookup = Array.isArray(layer.face_field_names) ? layer.face_field_names : [];
  const faceFieldValues = layer.face_field_values || null;
  if (Array.isArray(faceFieldValues)) {
    faceFieldNamesForLookup.forEach((name, idx) => {
      const values = faceFieldValues[idx];
      if (values) entry.faceFieldLookup.set(String(name), values);
    });
  }
  if (entry.mesh) entry.mesh.visible = entry.visible;
  if (entry.interactionProxy) entry.interactionProxy.visible = false;
  if (entry.maskedMesh) entry.maskedMesh.visible = false;
  if (entry.points) entry.points.visible = entry.visible;
  if (entry.maskedPoints) entry.maskedPoints.visible = false;
  updateEntryOpacity(entry, entry.opacity);
  updateEntryEdgeThickness(entry, entry.edgeThickness);
  if (entry.points) {
    entry.points.material.uniforms.size.value = entry.pointSize;
  }
  if (entry.maskedPoints) {
    entry.maskedPoints.material.uniforms.size.value = entry.pointSize;
  }
  if (entry.showVertices) {
    ensureMeshVertexPoints(entry);
  }
  updateEntryEdgeOverlay(entry);
  const fieldNames = layer.field_names || null;
  let didRenderColors = false;
  if (fieldNames && fieldNames.length > 0) {
    setLayerField(entry, Math.max(0, Math.min(entry.activeFieldIdx, fieldNames.length - 1)));
    didRenderColors = true;
  } else if (fieldValues || layer.values) {
    renderEntryColors(entry, entry.activeFieldIdx);
    didRenderColors = true;
  }
  if (!didRenderColors) {
    applyFiltersToEntry(entry);
  }
  } catch (error) {
    entry.loadError = error instanceof Error ? error.message : String(error);
    entry.visible = false;
    setLoadingStatus(`Failed to load ${entry.layer.label}: ${entry.loadError}`, { error: true });
    throw error;
  } finally {
    entry.loading = false;
    rebuildLayerList();
    if (!entry.loadError) setLoadingStatus('', { hidden: true });
  }
}

payload.layers.forEach((rawLayer, layerIdx) => {
  layers.push(makeLayerPlaceholder(rawLayer, layerIdx));
});

for (const entry of layers) {
  if (entry.visible) await loadLayerEntry(entry);
}
setLoadingStatus('', { hidden: true });

const toggleAllLayersInput = document.getElementById('toggle-all-layers');
if (toggleAllLayersInput) {
  toggleAllLayersInput.addEventListener('change', () => {
    void setAllLayersVisibility(toggleAllLayersInput.checked);
  });
}

initFilterPanelEvents();
rebuildLayerList();
refreshEdgeOverlay();

document.getElementById('btn-theme-mode').addEventListener('click', () => {
  applyThemeMode(!isDayMode);
});
const captureButton = document.getElementById('btn-capture');
captureButton.addEventListener('click', async () => {
  if (captureButton.disabled) return;
  captureButton.disabled = true;
  captureButton.classList.remove('captured');
  captureButton.classList.add('capturing');
  try {
    const saved = await captureCurrentView();
    if (saved) {
      captureButton.classList.remove('capturing');
      void captureButton.offsetWidth;
      captureButton.classList.add('captured');
      window.setTimeout(() => captureButton.classList.remove('captured'), 700);
    }
  } catch (error) {
    if (error?.name !== 'AbortError') {
      console.error('Failed to capture viewer image', error);
      window.alert(`Could not save screenshot: ${error}`);
    }
  } finally {
    captureButton.classList.remove('capturing');
    captureButton.disabled = false;
  }
});
const commandsToggleBtn = document.getElementById('btn-toggle-commands');
const controlsPanel = document.getElementById('controls');
if (commandsToggleBtn && controlsPanel) {
  const setCommandsVisible = (nextVisible) => {
    commandsVisible = Boolean(nextVisible);
    controlsPanel.classList.toggle('collapsed', !commandsVisible);
    commandsToggleBtn.textContent = commandsVisible ? 'Hide Commands' : 'Show Commands';
    commandsToggleBtn.classList.toggle('active', !commandsVisible);
  };
  const updateCommandsToggleButton = () => {
    setCommandsVisible(commandsVisible);
  };
  commandsToggleBtn.addEventListener('click', () => {
    setCommandsVisible(!commandsVisible);
  });
  if (controlsCloseButton) {
    controlsCloseButton.addEventListener('click', () => {
      setCommandsVisible(false);
    });
  }
  updateCommandsToggleButton();
  controlsPanel.addEventListener('scroll', () => {
    closeAllDropdownMenus();
  }, { passive: true });
}
document.getElementById('btn-reset-camera').addEventListener('click', () => {
  fitCamera();
});
let cameraInteractionActive = false;
let cameraInteractionEndTimer = 0;

function beginCameraInteraction() {
  clearTimeout(cameraInteractionEndTimer);
  cameraInteractionActive = true;
  for (const entry of layers) {
    if (!entry.visible || !entry.mesh) continue;
    const faceCount = Number(entry.layer.face_count) || 0;
    if (faceCount <= MESH_INTERACTION_PROXY_FACE_THRESHOLD) continue;
    const proxy = ensureMeshInteractionProxy(entry);
    if (!proxy) continue;
    entry.mesh.visible = false;
    if (entry.maskedMesh) entry.maskedMesh.visible = false;
    if (entry.edgeLines) entry.edgeLines.visible = false;
    proxy.visible = true;
  }
  requestRender();
}

function finishCameraInteraction() {
  cameraInteractionActive = false;
  for (const entry of layers) {
    if (entry.interactionProxy) entry.interactionProxy.visible = false;
    if (entry.mesh) entry.mesh.visible = entry.visible;
    updateMaskedObjectVisibility(entry);
    updateEntryEdgeOverlay(entry);
  }
  requestRender();
}

function scheduleCameraInteractionEnd(delay = 120) {
  clearTimeout(cameraInteractionEndTimer);
  cameraInteractionEndTimer = window.setTimeout(finishCameraInteraction, delay);
}

async function captureCurrentView() {
  if (cameraInteractionActive) finishCameraInteraction();
  const titleSlug = String(payload.title || 'geoviz')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'geoviz';
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${titleSlug}-${timestamp}.webp`;

  let fileHandle = null;
  if (typeof window.showSaveFilePicker === 'function') {
    fileHandle = await window.showSaveFilePicker({
      suggestedName: filename,
      startIn: 'downloads',
      types: [{
        description: 'WebP image',
        accept: { 'image/webp': ['.webp'] },
      }],
    });
  } else {
    const useDownload = window.confirm(
      'This browser cannot open the system Save dialog. Save the screenshot through browser downloads instead?',
    );
    if (!useDownload) return false;
  }

  const { width, height } = getViewerViewportSize();
  const context = renderer.getContext();
  const maxRenderbufferSize = context.getParameter(context.MAX_RENDERBUFFER_SIZE);
  const maxCapturePixels = 64_000_000;
  const captureScale = Math.max(
    1,
    Math.min(
      4,
      maxRenderbufferSize / width,
      maxRenderbufferSize / height,
      Math.sqrt(maxCapturePixels / (width * height)),
    ),
  );
  const captureWidth = Math.max(1, Math.floor(width * captureScale));
  const captureHeight = Math.max(1, Math.floor(height * captureScale));
  const previousPixelRatio = renderer.getPixelRatio();
  let imageUrl;
  try {
    renderer.setPixelRatio(1);
    renderer.setSize(captureWidth, captureHeight, false);
    refreshLineMaterialResolutions(captureWidth, captureHeight);
    controls.update();
    renderer.render(scene, camera);
    imageUrl = renderer.domElement.toDataURL('image/webp', 0.92);
  } finally {
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(previousPixelRatio);
    refreshLineMaterialResolutions(width, height);
    requestRender();
  }

  const imageBlob = await fetch(imageUrl).then((response) => response.blob());
  if (fileHandle) {
    const writable = await fileHandle.createWritable();
    await writable.write(imageBlob);
    await writable.close();
    return true;
  }

  const download = document.createElement('a');
  download.href = URL.createObjectURL(imageBlob);
  download.download = filename;
  document.body.appendChild(download);
  download.click();
  download.remove();
  window.setTimeout(() => URL.revokeObjectURL(download.href), 0);
  return true;
}

// Small-tile GPU depth picking avoids a linear CPU raycast (or a huge BVH
// build) on the multi-million-face atlas layers. Geometry and filter draw
// ranges are shared, not copied; the display itself remains full resolution.
// Padding keeps point centers inside the pick frustum when their sprites
// cover the cursor. A one-pixel frustum clips those centers before rasterizing.
const orbitPickSize = 33;
const orbitPickCenter = Math.floor(orbitPickSize / 2);
const orbitDepthTarget = new THREE.WebGLRenderTarget(orbitPickSize, orbitPickSize, {
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  depthBuffer: true, stencilBuffer: false,
});
const orbitDepthMaterial = new THREE.MeshDepthMaterial({
  depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide,
});
orbitDepthMaterial.toneMapped = false;
const orbitPointDepthMaterials = new WeakMap();
function orbitPointDepthMaterial(object) {
  let material = orbitPointDepthMaterials.get(object);
  if (!material) {
    material = new THREE.ShaderMaterial({
      uniforms: { size: { value: 1 } },
      vertexShader: `uniform float size;
        void main() {
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size;
        }`,
      fragmentShader: `#include <packing>
        void main() {
          vec2 center = gl_PointCoord - vec2(0.5);
          if (dot(center, center) > 0.25) discard;
          gl_FragColor = packDepthToRGBA(gl_FragCoord.z);
        }`,
      depthTest: true, depthWrite: true, toneMapped: false,
    });
    orbitPointDepthMaterials.set(object, material);
  }
  // Display shader sizes are drawing-buffer pixels; the pick tile uses CSS pixels.
  material.uniforms.size.value = object.material.uniforms.size.value / renderer.getPixelRatio();
  return material;
}
const orbitDepthPixels = new Uint8Array(orbitPickSize * orbitPickSize * 4);

// Sparse point sprites leave holes even over a visible surface. Prefer the
// frontmost hit in a small cursor neighborhood, rather than a background hit
// through the exact cursor pixel. Return its pixel as well as its depth so
// unprojection uses the ray that actually hit the surface.
function selectOrbitDepthPixel(pixels, size, radius = 6) {
  const center = Math.floor(size / 2);
  let best = null;
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const dx = col - center, dy = row - center;
      if (dx * dx + dy * dy > radius * radius) continue;
      const i = 4 * (row * size + col);
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2], a = pixels[i + 3];
      if (r === 255 && g === 255 && b === 255 && a === 255) continue;
      const depth = (r / 16777216 + g / 65536 + b / 256 + a) / 256;
      if (!best || depth < best.depth) best = { depth, dx, dy };
    }
  }
  return best;
}

function readOrbitSurfaceDepth(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const x = Math.floor(clientX - rect.left), y = Math.floor(clientY - rect.top);
  if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return null;
  const pickScene = new THREE.Scene();
  for (const entry of layers) {
    if (!entry.loaded || !entry.visible || !(entry.opacity > 0)) continue;
    for (const object of [entry.mesh, entry.maskedMesh]) {
      if (!object || (object === entry.maskedMesh && !object.visible)) continue;
      object.updateWorldMatrix(true, false);
      const mesh = new THREE.Mesh(object.geometry, orbitDepthMaterial);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(object.matrixWorld);
      pickScene.add(mesh);
    }
    for (const object of [entry.points, entry.maskedPoints]) {
      if (!object?.visible || !(object.material.uniforms.opacity.value > 0)) continue;
      object.updateWorldMatrix(true, false);
      const points = new THREE.Points(object.geometry, orbitPointDepthMaterial(object));
      points.matrixAutoUpdate = false;
      points.matrix.copy(object.matrixWorld);
      points.frustumCulled = false;
      pickScene.add(points);
    }
  }
  if (!pickScene.children.length) return null;
  const pickCamera = camera.clone();
  pickCamera.setViewOffset(rect.width, rect.height,
    x - orbitPickCenter, y - orbitPickCenter, orbitPickSize, orbitPickSize);
  const previousTarget = renderer.getRenderTarget();
  const previousViewport = renderer.getViewport(new THREE.Vector4());
  const previousScissor = renderer.getScissor(new THREE.Vector4());
  const previousScissorTest = renderer.getScissorTest();
  const previousClear = renderer.getClearColor(new THREE.Color());
  const previousAlpha = renderer.getClearAlpha();
  try {
    renderer.setRenderTarget(orbitDepthTarget);
    renderer.setViewport(0, 0, orbitPickSize, orbitPickSize);
    renderer.setScissorTest(false);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear();
    renderer.render(pickScene, pickCamera);
    renderer.readRenderTargetPixels(orbitDepthTarget, 0, 0, orbitPickSize, orbitPickSize, orbitDepthPixels);
  } finally {
    renderer.setRenderTarget(previousTarget);
    renderer.setViewport(previousViewport);
    renderer.setScissor(previousScissor);
    renderer.setScissorTest(previousScissorTest);
    renderer.setClearColor(previousClear, previousAlpha);
  }
  const hit = selectOrbitDepthPixel(orbitDepthPixels, orbitPickSize);
  if (!hit) return null;
  const { depth } = hit;
  const distance = camera.isPerspectiveCamera
    ? camera.near * camera.far / (camera.far - depth * (camera.far - camera.near))
    : camera.near + depth * (camera.far - camera.near);
  return Number.isFinite(distance) && distance > camera.near
    ? { distance, x: rect.left + x + hit.dx + 0.5,
        y: rect.top + y - hit.dy + 0.5 } : null;
}

function prepareSurfaceOrbit(event) {
  surfaceOrbitDragging = false;
  if (!controls.enabled || event.pointerType === 'touch' || event.button !== 0) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const hit = readOrbitSurfaceDepth(event.clientX, event.clientY);
  if (hit !== null) {
    const forward = camera.getWorldDirection(new THREE.Vector3());
    const ray = new THREE.Vector3(
      2 * (hit.x - rect.left) / rect.width - 1,
      1 - 2 * (hit.y - rect.top) / rect.height, 0.5,
    ).unproject(camera).sub(camera.position).normalize();
    surfaceOrbitPivot = camera.position.clone().addScaledVector(ray, hit.distance / ray.dot(forward));
  }
  // On empty background retain the last world-space anchor, not the target
  // translated by free dollying. No camera/target changes on pointer-down.
  if (!surfaceOrbitPivot) surfaceOrbitPivot = controls.target.clone();
  surfaceOrbitDragging = true;
}
// Capture runs before TrackballControls consumes the pointer-down event.
renderer.domElement.addEventListener('pointerdown', prepareSurfaceOrbit, { capture: true });
window.addEventListener('pointerup', () => { surfaceOrbitDragging = false; });
window.addEventListener('pointercancel', () => { surfaceOrbitDragging = false; });
window.addEventListener('blur', () => { surfaceOrbitDragging = false; });

function stopStuckCameraDrag() {
  surfaceOrbitDragging = false;
  const position = camera.position.clone();
  const up = camera.up.clone();
  const target = controls.target.clone();
  const zoom = camera.zoom;
  controls.reset();
  camera.position.copy(position);
  camera.up.copy(up);
  camera.zoom = zoom;
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.update();
}

renderer.domElement.addEventListener('pointermove', (event) => {
  if (cameraInteractionActive && event.pointerType === 'mouse' && event.buttons === 0) {
    stopStuckCameraDrag();
    scheduleCameraInteractionEnd(0);
  }
  // TrackballControls consumes movement in update(), not in pointermove.
  // Schedule frames during gestures only; idle movement does no picking/work.
  if (cameraInteractionActive && (event.buttons !== 0 || event.pointerType === 'touch')) {
    requestRender();
  }
});
renderer.domElement.addEventListener('wheel', () => {
  beginCameraInteraction();
  scheduleCameraInteractionEnd();
}, { passive: true });
renderer.domElement.addEventListener('pointerdown', beginCameraInteraction);
renderer.domElement.addEventListener('pointerup', () => scheduleCameraInteractionEnd());
window.addEventListener('pointerup', () => scheduleCameraInteractionEnd());
window.addEventListener('pointercancel', () => scheduleCameraInteractionEnd());

document.getElementById('btn-quit-top').addEventListener('click', async () => {
  try {
    await fetch('_shutdown', { method: 'POST' });
  } catch (_err) {
  }
  window.close();
  window.location.href = 'about:blank';
});

applyThemeMode(true);
function renderFrame() {
  renderQueued = false;
  controls.update();
  renderer.render(scene, camera);
}
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(renderFrame);
}
controls.addEventListener('change', requestRender);
syncViewerSize();
if (layers.length > 0) fitCamera();
requestRender();
