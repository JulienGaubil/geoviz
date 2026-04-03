import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import {
  acceleratedRaycast,
  computeBoundsTree,
  disposeBoundsTree,
} from 'three-mesh-bvh';

const payload = JSON.parse(document.getElementById('viewer-payload').textContent);
const colormapPresets = payload.colormap_presets || {};
const availableCmaps = payload.available_cmaps || Object.keys(colormapPresets);
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;
const arrayCtors = {
  float32: Float32Array,
  uint8: Uint8Array,
  uint32: Uint32Array,
  int32: Int32Array,
};

function decodeArraySpec(spec, fallbackCtor = Float32Array) {
  if (spec === null || spec === undefined) return null;
  if (ArrayBuffer.isView(spec)) return spec;
  if (spec && spec.encoding === 'base64' && typeof spec.data === 'string') {
    const ctor = arrayCtors[spec.dtype] || fallbackCtor;
    const binary = atob(spec.data);
    const bytes = new Uint8Array(binary.length);
    for (let idx = 0; idx < binary.length; idx += 1) {
      bytes[idx] = binary.charCodeAt(idx);
    }
    return new ctor(bytes.buffer);
  }
  if (!Array.isArray(spec)) {
    return null;
  }
  if (spec.length === 0) {
    return new fallbackCtor(0);
  }
  return new fallbackCtor(Array.isArray(spec[0]) ? spec.flat() : spec);
}

function normalizeLayerData(layer) {
  const kind = String(layer.layer_kind || (layer.points ? 'pointcloud' : 'mesh')).toLowerCase();
  layer.layer_kind = (kind === 'pointcloud') ? 'pointcloud' : 'mesh';
  layer.vertices = decodeArraySpec(layer.vertices, Float32Array) || new Float32Array(0);
  layer.faces = decodeArraySpec(layer.faces, Uint32Array) || new Uint32Array(0);
  layer.points = decodeArraySpec(layer.points, Float32Array) || new Float32Array(0);
  layer.edges = decodeArraySpec(layer.edges, Uint32Array);
  layer.colors = decodeArraySpec(layer.colors, Uint8Array) || new Uint8Array(0);
  layer.raw_colors = decodeArraySpec(layer.raw_colors, Uint8Array) || layer.colors;
  layer.values = decodeArraySpec(layer.values, Float32Array);
  layer.field_colors = Array.isArray(layer.field_colors)
    ? layer.field_colors.map((colors) => decodeArraySpec(colors, Uint8Array) || new Uint8Array(0))
    : null;
  layer.field_values = Array.isArray(layer.field_values)
    ? layer.field_values.map((values) => decodeArraySpec(values, Float32Array))
    : null;
  const fallbackVertexCount = layer.layer_kind === 'pointcloud'
    ? Math.floor(layer.points.length / 3)
    : Math.floor(layer.vertices.length / 3);
  layer.vertex_count = Number.isFinite(layer.vertex_count)
    ? Number(layer.vertex_count)
    : fallbackVertexCount;
  layer.face_count = Number.isFinite(layer.face_count)
    ? Number(layer.face_count)
    : Math.floor(layer.faces.length / 3);
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

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 1e7);
camera.position.set(0, 0, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x111827, 1.0);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 2.0;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.dynamicDampingFactor = 0.15;
controls.staticMoving = false;

const layers = [];
let isDayMode = false;
let commandsVisible = true;
const CLIP_SLIDER_MAX = 1000;
const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;
const pointerNdc = new THREE.Vector2();
const hoverTooltip = document.getElementById('hover-tooltip');
const dropdownPortal = document.getElementById('dropdown-portal');
const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
const hemisphereLight = new THREE.HemisphereLight(0xe0f2fe, 0x111827, 0.7);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
keyLight.position.set(2.5, 3.5, 4.0);
const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.4);
rimLight.position.set(-3.0, 1.5, -2.5);
scene.add(ambientLight, hemisphereLight, keyLight, rimLight);
const pointerState = {
  inside: false,
  clientX: 0,
  clientY: 0,
};
let hoverRefreshQueued = false;
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

function requestRender() {
  // Rendering is driven by the animation loop to match the point cloud viewer.
}

function scheduleHoverRefresh() {
  if (!pointerState.inside || hoverRefreshQueued) return;
  hoverRefreshQueued = true;
  requestAnimationFrame(() => {
    hoverRefreshQueued = false;
    updateHoverFromPointer();
  });
}

function requestRenderAndHover() {
  requestRender();
  scheduleHoverRefresh();
}

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

function getEntryFieldCount(entry) {
  return Math.max(
    Array.isArray(entry.layer.field_names) ? entry.layer.field_names.length : 0,
    Array.isArray(entry.layer.field_values) ? entry.layer.field_values.length : 0,
    Array.isArray(entry.layer.field_scalar_meta) ? entry.layer.field_scalar_meta.length : 0,
    Array.isArray(entry.layer.field_colors) ? entry.layer.field_colors.length : 0,
    1,
  );
}

function ensureFieldClipStates(entry) {
  const fieldCount = getEntryFieldCount(entry);
  if (!Array.isArray(entry.fieldClipStates)) {
    entry.fieldClipStates = [];
  }
  while (entry.fieldClipStates.length < fieldCount) {
    entry.fieldClipStates.push(createClipState());
  }
  return entry.fieldClipStates;
}

function getClipState(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  if (entry.shareFieldScale) {
    if (!entry.sharedFieldClip) {
      entry.sharedFieldClip = createClipState();
    }
    return entry.sharedFieldClip;
  }
  const states = ensureFieldClipStates(entry);
  const clampedIdx = Math.max(0, Math.min(states.length - 1, fieldIdx));
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

function collectScalarMetas(layer) {
  const metas = [];
  if (layer && layer.scalar_meta) metas.push(layer.scalar_meta);
  if (layer && Array.isArray(layer.field_scalar_meta)) {
    layer.field_scalar_meta.forEach((meta) => {
      if (meta) metas.push(meta);
    });
  }
  return metas;
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
  requestRender();
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

function setLineMaterialResolution(material) {
  if (!material || !material.resolution) return;
  material.resolution.set(Math.max(1, window.innerWidth), Math.max(1, window.innerHeight));
}

function refreshLineMaterialResolutions() {
  for (const entry of layers) {
    if (entry.edgeLines?.material) setLineMaterialResolution(entry.edgeLines.material);
    if (entry.pointLines?.material) setLineMaterialResolution(entry.pointLines.material);
  }
}

function buildMesh(layer) {
  const geometry = new THREE.BufferGeometry();
  const positionAttr = new THREE.BufferAttribute(layer.vertices, 3);
  positionAttr.setUsage(THREE.StaticDrawUsage);
  geometry.setAttribute('position', positionAttr);
  geometry.setAttribute('color', createStoredColorAttribute(layer.colors));
  const indexAttr = new THREE.BufferAttribute(layer.faces, 1);
  indexAttr.setUsage(THREE.StaticDrawUsage);
  geometry.setIndex(indexAttr);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  if (layer.face_count > 0) {
    geometry.computeBoundsTree();
  }

  const initialOpacity = Number(layer.opacity ?? payload.opacity) || 1.0;
  const material = new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: initialOpacity < 0.999,
    opacity: initialOpacity,
    depthWrite: initialOpacity >= 0.999,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = true;
  scene.add(mesh);

  const wireframeGeometry = new THREE.WireframeGeometry(geometry);
  const wirePositionsAttr = wireframeGeometry.getAttribute('position');
  const wirePositions = wirePositionsAttr ? Array.from(wirePositionsAttr.array) : [];
  wireframeGeometry.dispose();
  const edgeThickness = getLayerEdgeThickness(layer);
  const edgeMaterial = new LineMaterial({
    color: edgeColorHex(),
    transparent: true,
    opacity: isDayMode ? 0.92 : 1.0,
    depthTest: true,
    depthWrite: false,
    linewidth: edgeThickness,
  });
  setLineMaterialResolution(edgeMaterial);
  let edgeLines = null;
  if (wirePositions.length > 0) {
    const edgeGeometry = new LineSegmentsGeometry();
    edgeGeometry.setPositions(wirePositions);
    edgeLines = new LineSegments2(edgeGeometry, edgeMaterial);
    edgeLines.visible = false;
    edgeLines.frustumCulled = false;
    edgeLines.renderOrder = 3;
    scene.add(edgeLines);
  } else {
    edgeMaterial.dispose();
  }
  return { mesh, edgeLines };
}

function buildPoints(layer) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(layer.points, 3));
  geometry.setAttribute('color', createStoredColorAttribute(layer.colors));

  const pointSize = Math.min(12.0, Math.max(1.0, Number(layer.point_size ?? payload.point_size) || 5.0));
  const initialOpacity = Number(layer.opacity ?? payload.opacity) || 1.0;
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

  let lines = null;
  if (layer.edges && layer.edges.length >= 2) {
    const edgeThickness = getLayerEdgeThickness(layer);
    const positions = [];
    for (let idx = 0; idx + 1 < layer.edges.length; idx += 2) {
      const i = layer.edges[idx];
      const j = layer.edges[idx + 1];
      if (i < 0 || j < 0) continue;
      const iBase = i * 3;
      const jBase = j * 3;
      if (jBase + 2 >= layer.points.length || iBase + 2 >= layer.points.length) continue;
      positions.push(
        layer.points[iBase], layer.points[iBase + 1], layer.points[iBase + 2],
        layer.points[jBase], layer.points[jBase + 1], layer.points[jBase + 2],
      );
    }
    if (positions.length > 0) {
      const lineGeometry = new LineSegmentsGeometry();
      lineGeometry.setPositions(positions);
      const lineMaterial = new LineMaterial({
        color: isDayMode ? 0x0f172a : 0xe4e4e7,
        opacity: (isDayMode ? 0.28 : 0.35) * initialOpacity,
        transparent: true,
        linewidth: edgeThickness,
      });
      setLineMaterialResolution(lineMaterial);
      lines = new LineSegments2(lineGeometry, lineMaterial);
      lines.frustumCulled = false;
      scene.add(lines);
    }
  }
  return { points, lines };
}

function updateEntryOpacity(entry, opacity) {
  const nextOpacity = Math.min(1.0, Math.max(0.05, Number(opacity) || 1.0));
  entry.opacity = nextOpacity;
  if (entry.mesh) {
    entry.mesh.material.opacity = nextOpacity;
    entry.mesh.material.transparent = nextOpacity < 0.999;
    entry.mesh.material.depthWrite = nextOpacity >= 0.999;
    entry.mesh.material.needsUpdate = true;
  }
  if (entry.points) {
    entry.points.material.uniforms.opacity.value = nextOpacity;
  }
  if (entry.pointLines) {
    entry.pointLines.material.opacity = (isDayMode ? 0.28 : 0.35) * nextOpacity;
  }
}

function updateEntryEdgeThickness(entry, thickness) {
  const nextThickness = Math.min(12.0, Math.max(1.0, Number(thickness) || 1.0));
  entry.edgeThickness = nextThickness;
  if (entry.edgeLines) {
    entry.edgeLines.material.linewidth = nextThickness;
  }
  if (entry.pointLines) {
    entry.pointLines.material.linewidth = nextThickness;
  }
}

function updateEntryEdgeOverlay(entry) {
  if (entry.edgeLines) {
    const color = edgeColorHex();
    entry.edgeLines.visible = entry.visible && entry.showEdges;
    entry.edgeLines.material.color.setHex(color);
    entry.edgeLines.material.opacity = isDayMode ? 0.82 : 0.92;
  }
  if (entry.pointLines) {
    entry.pointLines.visible = entry.visible;
    entry.pointLines.material.color.setHex(isDayMode ? 0x0f172a : 0xe4e4e7);
    entry.pointLines.material.opacity = (isDayMode ? 0.28 : 0.35) * entry.opacity;
  }
}

function refreshEdgeOverlay() {
  for (const entry of layers) {
    updateEntryEdgeOverlay(entry);
  }
  requestRender();
}

const NO_CMAP = 'none';

function getFallbackCmap() {
  return availableCmaps[0] || 'viridis';
}

function isNoCmap(cmapName) {
  return !cmapName || cmapName === NO_CMAP;
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

function getEntryDisplayedValues(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const fieldValues = entry.layer.field_values || null;
  return fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
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

function getEntryRawColors(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const fieldColors = entry.layer.field_colors || null;
  if (fieldColors && fieldIdx >= 0 && fieldIdx < fieldColors.length && fieldColors[fieldIdx]) {
    return fieldColors[fieldIdx];
  }
  return entry.layer.raw_colors || entry.layer.colors || null;
}

function getEntryColorModeOptions(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const options = [NO_CMAP];
  const values = getEntryDisplayedValues(entry, fieldIdx);
  if (values && typeof values.length === 'number') {
    availableCmaps.forEach((cmapName) => options.push(cmapName));
  }
  return Array.from(new Set(options));
}

function getEntryRawGradient(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const values = getEntryDisplayedValues(entry, fieldIdx);
  const valueRange = computeFiniteValueRange(values);
  if (valueRange) return buildRawScalarGradient(valueRange);
  return sampleRawGradientFromColors(getEntryRawColors(entry, fieldIdx));
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

function createCmapSwatch(entry, fieldIdx, cmapName) {
  const swatch = document.createElement('span');
  swatch.className = 'cmap-swatch';
  const gradient = isNoCmap(cmapName)
    ? getEntryRawGradient(entry, fieldIdx)
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
  value.textContent = isNoCmap(cmapName) ? NO_CMAP : cmapName;
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

function getEntryCmap(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  return entry.activeCmap || NO_CMAP;
}

function getEntryScaleRange(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const values = getEntryDisplayedValues(entry, fieldIdx);
  const fieldMetas = entry.layer.field_scalar_meta || null;
  const baseMeta = fieldMetas ? fieldMetas[fieldIdx] : entry.layer.scalar_meta;
  const metaMin = Number(baseMeta && baseMeta.min);
  const metaMax = Number(baseMeta && baseMeta.max);
  if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {
    return { min: metaMin, max: metaMax };
  }
  return computeFiniteValueRange(values);
}

function getEntryAllScaleRanges(entry) {
  const ranges = [];
  const fieldValues = entry.layer.field_values || null;
  const fieldMetas = entry.layer.field_scalar_meta || null;
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
  const range = getEntryScaleRange(entry);
  return range ? [range] : [];
}

function getSharedFieldScaleRange(entry) {
  if (!entry || isNoCmap(getEntryCmap(entry))) return null;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (const range of getEntryAllScaleRanges(entry)) {
    minVal = Math.min(minVal, range.min);
    maxVal = Math.max(maxVal, range.max);
  }
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  return { min: minVal, max: maxVal };
}

function getScalarMeta(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const values = getEntryDisplayedValues(entry, fieldIdx);
  const fieldMetas = entry.layer.field_scalar_meta || null;
  const baseMeta = fieldMetas ? fieldMetas[fieldIdx] : entry.layer.scalar_meta;
  const cmapName = getEntryCmap(entry, fieldIdx);
  if (!baseMeta) {
    const range = computeFiniteValueRange(values);
    if (!range) return null;
    const resolved = {
      min: range.min,
      max: range.max,
      cmap: cmapName,
      gradient: isNoCmap(cmapName)
        ? getEntryRawGradient(entry, fieldIdx)
        : (getColormapPreset(cmapName)?.gradient || null),
    };
    const sharedFieldScaleRange = getSharedFieldScaleRange(entry);
    if (!isNoCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {
      resolved.min = sharedFieldScaleRange.min;
      resolved.max = sharedFieldScaleRange.max;
    }
    resolved.fullMin = resolved.min;
    resolved.fullMax = resolved.max;
    if (!isNoCmap(cmapName)) {
      const clippedRange = applyClipToRange(
        { min: resolved.fullMin, max: resolved.fullMax },
        getClipState(entry, fieldIdx),
      );
      if (clippedRange) {
        resolved.min = clippedRange.min;
        resolved.max = clippedRange.max;
      }
    }
    resolved.clipState = createClipState(
      getClipState(entry, fieldIdx).minFrac,
      getClipState(entry, fieldIdx).maxFrac,
    );
    resolved.clipEnabled = !isNoCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
    return resolved;
  }
  const resolved = {
    ...baseMeta,
    cmap: cmapName,
    gradient: isNoCmap(cmapName)
      ? (baseMeta.gradient || getEntryRawGradient(entry, fieldIdx))
      : (getColormapPreset(cmapName)?.gradient || baseMeta.gradient),
  };
  const sharedFieldScaleRange = getSharedFieldScaleRange(entry);
  if (!isNoCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {
    resolved.min = sharedFieldScaleRange.min;
    resolved.max = sharedFieldScaleRange.max;
  }
  resolved.fullMin = resolved.min;
  resolved.fullMax = resolved.max;
  if (!isNoCmap(cmapName)) {
    const clippedRange = applyClipToRange(
      { min: resolved.fullMin, max: resolved.fullMax },
      getClipState(entry, fieldIdx),
    );
    if (clippedRange) {
      resolved.min = clippedRange.min;
      resolved.max = clippedRange.max;
    }
  }
  resolved.clipState = createClipState(
    getClipState(entry, fieldIdx).minFrac,
    getClipState(entry, fieldIdx).maxFrac,
  );
  resolved.clipEnabled = !isNoCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
  return resolved;
}

function applyStoredColors(entry, colors) {
  if (!entry.colorGeometry) return;
  setStoredColorAttribute(entry.colorGeometry, colors);
}

function applyColorsFromValues(entry, values, meta) {
  const preset = getColormapPreset(meta && meta.cmap);
  const samples = preset && Array.isArray(preset.samples) ? preset.samples : null;
  if (!samples || samples.length === 0 || !values || typeof values.length !== 'number') return false;

  const colorCount = values.length * 3;
  if (!entry.colorGeometry) return false;
  const existing = entry.colorGeometry.getAttribute('color');
  const colors = (
    existing
    && existing.array instanceof Uint8Array
    && existing.array.length === colorCount
    && existing.normalized
  )
    ? existing.array
    : new Uint8Array(colorCount);
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

  if (
    existing
    && existing.array instanceof Uint8Array
    && existing.array.length === colorCount
    && existing.normalized
  ) {
    existing.needsUpdate = true;
    return true;
  }

  entry.colorGeometry.setAttribute('color', createStoredColorAttribute(colors));
  return true;
}

function renderEntryColors(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
  const fieldValues = entry.layer.field_values || null;
  const values = fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
  const rawColors = getEntryRawColors(entry, fieldIdx);
  const cmapName = getEntryCmap(entry, fieldIdx);
  const meta = getScalarMeta(entry, fieldIdx);
  const updatedFromValues = !isNoCmap(cmapName) && values && meta ? applyColorsFromValues(entry, values, meta) : false;

  if (updatedFromValues) {
    entry.colorGeometry?.attributes?.color && (entry.colorGeometry.attributes.color.needsUpdate = true);
  } else if (rawColors) {
    applyStoredColors(entry, rawColors);
  } else {
    applyStoredColors(entry, entry.layer.colors);
  }

  entry.activeFieldIdx = fieldIdx;
  entry.activeValues = values;
  entry.activeCmap = cmapName;
  requestRenderAndHover();
}

function setLayerCmap(entry, cmapName) {
  const fieldIdx = entry.activeFieldIdx ?? 0;
  entry.activeCmap = cmapName;
  renderEntryColors(entry, fieldIdx);
  updateColorbars();
}

function setLayerShareFieldScale(entry, enabled) {
  entry.shareFieldScale = Boolean(enabled);
  const fieldIdx = Math.max(
    0,
    Math.min(
      entry.activeFieldIdx ?? 0,
      entry.fieldCmaps && entry.fieldCmaps.length > 0 ? entry.fieldCmaps.length - 1 : 0,
    ),
  );
  renderEntryColors(entry, fieldIdx);
  updateColorbars();
  hideHoverTooltip();
}

function setColorClipBoundary(entry, fieldIdx, boundary, value) {
  const state = getClipState(entry, fieldIdx);
  setClipBoundary(state, boundary, sliderValueToClip(value, boundary === 'max' ? 1.0 : 0.0));
  renderEntryColors(entry, fieldIdx);
  syncAllColorbarClipControls();
  hideHoverTooltip();
}

function fitCamera() {
  const box = new THREE.Box3();
  let hasBox = false;
  for (const entry of layers) {
    if (!entry.visible) continue;
    if (!entry.mainObject) continue;
    box.expandByObject(entry.mainObject);
    hasBox = true;
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
  controls.update();
  requestRenderAndHover();
}

function createEntryFieldPanel(entry) {
  const fieldIdx = entry.activeFieldIdx ?? 0;
  const fieldNames = entry.layer.field_names || null;
  const meta = getScalarMeta(entry, fieldIdx);
  if (!meta) return null;

  const fieldLabel = fieldNames ? fieldNames[fieldIdx] : null;
  const colorbarLabel = fieldLabel ? `Field: ${fieldLabel}` : 'Field colors';
  const controlsEnabled = entry.visible;

  const wrapper = document.createElement('div');
  wrapper.className = 'colorbar';
  const title = document.createElement('div');
  title.className = 'colorbar-title';
  title.textContent = colorbarLabel;
  const dropdownWrap = document.createElement('div');
  dropdownWrap.className = 'layer-select-wrapper';
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-dropdown';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'layer-select-button cmap-dropdown-button';
  button.disabled = !controlsEnabled;
  const currentCmap = getEntryCmap(entry, fieldIdx);
  setCmapButtonContent(button, currentCmap);
  const menu = document.createElement('div');
  menu.className = 'custom-dropdown-menu';
  getEntryColorModeOptions(entry, fieldIdx).forEach((cmapName) => {
    const item = document.createElement('div');
    item.className = 'custom-dropdown-item cmap-dropdown-item';
    if (!controlsEnabled) item.classList.add('disabled');
    if (cmapName === currentCmap) item.classList.add('selected');
    const prefix = document.createElement('span');
    prefix.className = 'cmap-dropdown-prefix';
    prefix.appendChild(createCmapSwatch(entry, fieldIdx, cmapName));
    item.appendChild(prefix);
    const value = document.createElement('span');
    value.className = 'layer-id';
    value.textContent = isNoCmap(cmapName) ? NO_CMAP : cmapName;
    item.appendChild(value);
    item.addEventListener('click', () => {
      if (!controlsEnabled) return;
      setLayerCmap(entry, cmapName);
      setCmapButtonContent(button, cmapName);
      menu.querySelectorAll('.custom-dropdown-item').forEach((el) => {
        el.classList.toggle('selected', el === item);
      });
      closeDropdownMenu(menu);
    });
    menu.appendChild(item);
  });
  button.addEventListener('click', (ev) => {
    if (!controlsEnabled) return;
    ev.stopPropagation();
    toggleDropdownMenu(button, menu);
  });
  const visual = document.createElement('div');
  visual.className = 'colorbar-visual';
  const distributionWrap = document.createElement('div');
  distributionWrap.className = 'colorbar-distribution-wrap';
  const distributionViz = buildDistributionSvg(
    getEntryDisplayedValues(entry, fieldIdx),
    { min: meta.fullMin, max: meta.fullMax },
  );
  const distribution = distributionViz ? distributionViz.svg : null;
  const gradientWrap = document.createElement('div');
  gradientWrap.className = 'colorbar-gradient-wrap';
  const gradient = document.createElement('div');
  gradient.className = 'colorbar-gradient';
  gradient.style.background = gradientCss(meta.gradient);
  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  dropdownWrap.appendChild(dropdown);
  wrapper.appendChild(title);
  wrapper.appendChild(dropdownWrap);
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

    const currentCmapIsClipped = !isNoCmap(currentCmap);
    if (currentCmapIsClipped) {
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
        const currentMeta = getScalarMeta(entry, fieldIdx);
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
        const currentMeta = getScalarMeta(entry, fieldIdx);
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
        setColorClipBoundary(entry, fieldIdx, 'min', minInput.value);
      });
      maxInput.addEventListener('input', () => {
        setColorClipBoundary(entry, fieldIdx, 'max', maxInput.value);
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
    if (!entry.fieldOptionsContainer) continue;
    entry.fieldOptionsContainer.innerHTML = '';
    const panel = createEntryFieldPanel(entry);
    if (panel) entry.fieldOptionsContainer.appendChild(panel);
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

function setLayerField(entry, fieldIdx) {
  const fieldNames = entry.layer.field_names || null;
  if (!fieldNames || fieldIdx < 0 || fieldIdx >= fieldNames.length) return;
  renderEntryColors(entry, fieldIdx);
  updateColorbars();
}

function hideHoverTooltip() {
  hoverTooltip.style.display = 'none';
}

function toHexChannel(value) {
  return Math.round(Math.min(255, Math.max(0, value))).toString(16).padStart(2, '0');
}

function formatHoverColor(colorAttr, idx) {
  if (!colorAttr || idx < 0 || idx >= colorAttr.count) return null;
  let r = colorAttr.getX(idx);
  let g = colorAttr.getY(idx);
  let b = colorAttr.getZ(idx);
  if (![r, g, b].every(Number.isFinite)) return null;
  const red = Math.round(Math.min(1, Math.max(0, r)) * 255);
  const green = Math.round(Math.min(1, Math.max(0, g)) * 255);
  const blue = Math.round(Math.min(1, Math.max(0, b)) * 255);
  return {
    rgb: `rgb(${red}, ${green}, ${blue})`,
    hex: `#${toHexChannel(red)}${toHexChannel(green)}${toHexChannel(blue)}`,
  };
}

function buildHoverTooltipContent(details) {
  hoverTooltip.replaceChildren();

  const title = document.createElement('div');
  title.className = 'hover-tooltip-title';
  title.textContent = details.layerLabel;
  hoverTooltip.appendChild(title);

  const addRow = (label, value = null, extraNode = null) => {
    const row = document.createElement('div');
    row.className = 'hover-tooltip-row';

    const labelNode = document.createElement('div');
    labelNode.className = 'hover-tooltip-label';
    labelNode.textContent = label;
    row.appendChild(labelNode);

    if (extraNode) {
      row.appendChild(extraNode);
    }

    if (value !== null && value !== undefined && value !== '') {
      const valueNode = document.createElement('div');
      valueNode.className = 'hover-tooltip-value';
      valueNode.textContent = value;
      row.appendChild(valueNode);
    }

    hoverTooltip.appendChild(row);
  };

  addRow('xyz', details.xyz);
  addRow('color', null, (() => {
    const swatch = document.createElement('div');
    swatch.className = 'hover-tooltip-swatch';
    swatch.style.background = details.color.rgb;
    return swatch;
  })());
  addRow('value', details.valueText);
}

function showHoverTooltip(clientX, clientY, details) {
  buildHoverTooltipContent(details);
  hoverTooltip.style.display = 'block';
  const margin = 12;
  const maxX = window.innerWidth - hoverTooltip.offsetWidth - margin;
  const maxY = window.innerHeight - hoverTooltip.offsetHeight - margin;
  hoverTooltip.style.left = `${Math.min(clientX + 14, Math.max(margin, maxX))}px`;
  hoverTooltip.style.top = `${Math.min(clientY + 14, Math.max(margin, maxY))}px`;
}

function pickClosestFaceVertexIndex(hit, mesh, posAttr) {
  if (!hit || !hit.face || !mesh || !posAttr) return -1;
  const localHit = mesh.worldToLocal(hit.point.clone());
  const candidates = [hit.face.a, hit.face.b, hit.face.c];
  let bestIdx = -1;
  let bestDist2 = Infinity;
  for (const idx of candidates) {
    if (idx < 0 || idx >= posAttr.count) continue;
    const dx = posAttr.getX(idx) - localHit.x;
    const dy = posAttr.getY(idx) - localHit.y;
    const dz = posAttr.getZ(idx) - localHit.z;
    const dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 < bestDist2) {
      bestDist2 = dist2;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

function getEntryPointSize(entry) {
  if (!entry) return 1.0;
  const numeric = Number(entry.pointSize ?? entry.layer?.point_size ?? payload.point_size);
  return Number.isFinite(numeric) ? Math.min(12.0, Math.max(1.0, numeric)) : 1.0;
}

function computePointHoverThreshold(viewportHeightPx, entry) {
  const distance = camera.position.distanceTo(controls.target);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const worldPerPixel = (2 * Math.max(distance, 1e-6) * Math.tan(fov / 2)) / Math.max(viewportHeightPx, 1);
  const pointRadiusPx = Math.max(getEntryPointSize(entry) * 0.5, 1.5);
  return Math.max(worldPerPixel * pointRadiusPx * 1.15, 1e-5);
}

function pickBestPointHit(hits) {
  if (!hits || hits.length === 0) return null;
  let bestHit = null;
  for (const hit of hits) {
    if (hit.index === undefined || hit.index === null) continue;
    if (!bestHit) {
      bestHit = hit;
      continue;
    }
    const bestDistanceToRay = Number.isFinite(bestHit.distanceToRay) ? bestHit.distanceToRay : Infinity;
    const hitDistanceToRay = Number.isFinite(hit.distanceToRay) ? hit.distanceToRay : Infinity;
    if (hitDistanceToRay < bestDistanceToRay - 1e-9) {
      bestHit = hit;
      continue;
    }
    if (Math.abs(hitDistanceToRay - bestDistanceToRay) <= 1e-9 && hit.distance < bestHit.distance) {
      bestHit = hit;
    }
  }
  return bestHit;
}

function updateHoverFromPointer() {
  if (!pointerState.inside) {
    hideHoverTooltip();
    return;
  }
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (pointerState.clientX - rect.left) / rect.width;
  const y = (pointerState.clientY - rect.top) / rect.height;
  pointerNdc.x = x * 2 - 1;
  pointerNdc.y = -(y * 2 - 1);
  raycaster.setFromCamera(pointerNdc, camera);

  let best = null;
  for (const entry of layers) {
    if (!entry.visible) continue;
    let hit = null;
    if (entry.points) {
      raycaster.params.Points.threshold = computePointHoverThreshold(rect.height, entry);
      const hits = raycaster.intersectObject(entry.points, false);
      hit = pickBestPointHit(hits);
    } else if (entry.mesh) {
      const hits = raycaster.intersectObject(entry.mesh, false);
      hit = (hits && hits.length > 0) ? hits[0] : null;
    }
    if (!hit) continue;
    if (!best || hit.distance < best.hit.distance) {
      best = { entry, hit };
    }
  }

  if (!best) {
    hideHoverTooltip();
    return;
  }

  const posAttr = best.entry.colorGeometry?.getAttribute('position');
  const colorAttr = best.entry.colorGeometry?.getAttribute('color');
  let idx = -1;
  if (best.entry.points) {
    idx = Number(best.hit.index ?? -1);
  } else if (best.entry.mesh) {
    idx = pickClosestFaceVertexIndex(best.hit, best.entry.mesh, posAttr);
  }
  if (idx < 0) {
    hideHoverTooltip();
    return;
  }

  const px = posAttr.getX(idx);
  const py = posAttr.getY(idx);
  const pz = posAttr.getZ(idx);
  const hoverColor = formatHoverColor(colorAttr, idx);
  if (!hoverColor) {
    hideHoverTooltip();
    return;
  }

  const fieldNames = best.entry.layer.field_names || null;
  const fieldLabel = fieldNames ? fieldNames[best.entry.activeFieldIdx ?? 0] : null;
  const activeValues = best.entry.activeValues || best.entry.layer.values || null;
  const value = activeValues && idx < activeValues.length ? activeValues[idx] : null;
  const valueText = (value === null || value === undefined || !Number.isFinite(value))
    ? 'n/a'
    : Number(value).toFixed(4);

  showHoverTooltip(pointerState.clientX, pointerState.clientY, {
    layerLabel: `${best.entry.layer.label}${fieldLabel ? ` (${fieldLabel})` : ''}`,
    xyz: `(${px.toFixed(4)}, ${py.toFixed(4)}, ${pz.toFixed(4)})`,
    color: hoverColor,
    valueText: valueText,
  });
}

function rebuildLayerList() {
  const container = document.getElementById('layer-list');
  container.innerHTML = '';
  layers.forEach((entry, idx) => {
    const row = document.createElement('div');
    row.className = 'layer-row';
    const isPointLayer = entry.layerKind === 'pointcloud';
    const hasMultipleFields = getEntryFieldCount(entry) > 1;
    const hasFieldOptions = true;
    const hasScalarOptions = getEntryFieldCount(entry) > 0 || Boolean(entry.layer.scalar_meta) || Boolean(entry.layer.values);
    let refreshFieldControls = null;
    let controlsBody = null;
    let toggleButton = null;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = entry.visible;
    checkbox.addEventListener('change', () => {
      entry.visible = checkbox.checked;
      if (entry.mesh) entry.mesh.visible = entry.visible;
      if (entry.points) entry.points.visible = entry.visible;
      if (entry.pointLines) entry.pointLines.visible = entry.visible;
      updateEntryEdgeOverlay(entry);
      rebuildLayerList();
      hideHoverTooltip();
      requestRender();
    });

    const info = document.createElement('div');
    info.className = 'layer-main';
    const label = document.createElement('div');
    label.className = 'layer-label';
    label.textContent = `${idx + 1}. ${entry.layer.label}`;
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
      const pointSizeWrap = document.createElement('div');
      pointSizeWrap.className = 'layer-option-control';
      const pointSizeLabel = document.createElement('label');
      pointSizeLabel.textContent = 'Point size';
      const pointSizeInput = document.createElement('input');
      pointSizeInput.type = 'range';
      pointSizeInput.min = '1';
      pointSizeInput.max = '12';
      pointSizeInput.step = '0.5';
      pointSizeInput.value = String(entry.pointSize);
      pointSizeInput.disabled = !entry.visible;
      pointSizeInput.addEventListener('input', () => {
        const value = Math.min(12.0, Math.max(1.0, Number(pointSizeInput.value) || entry.pointSize));
        entry.pointSize = value;
        if (entry.points) {
          entry.points.material.uniforms.size.value = value;
        }
        requestRenderAndHover();
      });
      pointSizeWrap.appendChild(pointSizeLabel);
      pointSizeWrap.appendChild(pointSizeInput);
      optionGrid.appendChild(pointSizeWrap);

      const opacityWrap = document.createElement('div');
      opacityWrap.className = 'layer-option-control';
      const opacityLabel = document.createElement('label');
      opacityLabel.textContent = 'Point opacity';
      const opacityInput = document.createElement('input');
      opacityInput.type = 'range';
      opacityInput.min = '0.05';
      opacityInput.max = '1';
      opacityInput.step = '0.05';
      opacityInput.value = String(entry.opacity);
      opacityInput.disabled = !entry.visible;
      opacityInput.addEventListener('input', () => {
        updateEntryOpacity(entry, opacityInput.value);
        requestRenderAndHover();
      });
      opacityWrap.appendChild(opacityLabel);
      opacityWrap.appendChild(opacityInput);
      optionGrid.appendChild(opacityWrap);

      if (entry.pointLines && entry.visible) {
        const edgeThicknessWrap = document.createElement('div');
        edgeThicknessWrap.className = 'layer-option-control';
        const edgeThicknessLabel = document.createElement('label');
        edgeThicknessLabel.textContent = 'Edge thickness';
        const edgeThicknessInput = document.createElement('input');
        edgeThicknessInput.type = 'range';
        edgeThicknessInput.min = '1';
        edgeThicknessInput.max = '12';
        edgeThicknessInput.step = '0.5';
        edgeThicknessInput.value = String(entry.edgeThickness);
        edgeThicknessInput.disabled = !entry.visible;
        edgeThicknessInput.addEventListener('input', () => {
          updateEntryEdgeThickness(entry, edgeThicknessInput.value);
          requestRenderAndHover();
        });
        edgeThicknessWrap.appendChild(edgeThicknessLabel);
        edgeThicknessWrap.appendChild(edgeThicknessInput);
        optionGrid.appendChild(edgeThicknessWrap);
      }
    } else {
      const opacityWrap = document.createElement('div');
      opacityWrap.className = 'layer-option-control';
      const opacityLabel = document.createElement('label');
      opacityLabel.textContent = 'Mesh opacity';
      const opacityInput = document.createElement('input');
      opacityInput.type = 'range';
      opacityInput.min = '0.05';
      opacityInput.max = '1';
      opacityInput.step = '0.05';
      opacityInput.value = String(entry.opacity);
      opacityInput.disabled = !entry.visible;
      opacityInput.addEventListener('input', () => {
        updateEntryOpacity(entry, opacityInput.value);
        requestRenderAndHover();
      });
      opacityWrap.appendChild(opacityLabel);
      opacityWrap.appendChild(opacityInput);
      optionGrid.appendChild(opacityWrap);

      const edgeToggleLabel = document.createElement('label');
      edgeToggleLabel.className = 'layer-option-toggle';
      const edgeToggleInput = document.createElement('input');
      edgeToggleInput.type = 'checkbox';
      edgeToggleInput.checked = Boolean(entry.showEdges);
      edgeToggleInput.disabled = !entry.visible;
      edgeToggleInput.addEventListener('change', () => {
        entry.showEdges = edgeToggleInput.checked;
        updateEntryEdgeOverlay(entry);
        rebuildLayerList();
      });
      edgeToggleLabel.appendChild(edgeToggleInput);
      edgeToggleLabel.appendChild(document.createTextNode('Show mesh edges'));
      optionGrid.appendChild(edgeToggleLabel);

      if (entry.edgeLines && entry.showEdges && entry.visible) {
        const edgeThicknessWrap = document.createElement('div');
        edgeThicknessWrap.className = 'layer-option-control';
        const edgeThicknessLabel = document.createElement('label');
        edgeThicknessLabel.textContent = 'Edge thickness';
        const edgeThicknessInput = document.createElement('input');
        edgeThicknessInput.type = 'range';
        edgeThicknessInput.min = '1';
        edgeThicknessInput.max = '12';
        edgeThicknessInput.step = '0.5';
        edgeThicknessInput.value = String(entry.edgeThickness);
        edgeThicknessInput.disabled = !entry.visible;
        edgeThicknessInput.addEventListener('input', () => {
          updateEntryEdgeThickness(entry, edgeThicknessInput.value);
          requestRenderAndHover();
        });
        edgeThicknessWrap.appendChild(edgeThicknessLabel);
        edgeThicknessWrap.appendChild(edgeThicknessInput);
        optionGrid.appendChild(edgeThicknessWrap);
      }
    }

    if (hasMultipleFields) {
      const shareScaleLabel = document.createElement('label');
      shareScaleLabel.className = 'layer-option-toggle';
      const shareScaleInput = document.createElement('input');
      shareScaleInput.type = 'checkbox';
      shareScaleInput.checked = Boolean(entry.shareFieldScale);
      shareScaleInput.disabled = !entry.visible || !getSharedFieldScaleRange(entry);
      shareScaleInput.addEventListener('change', () => {
        setLayerShareFieldScale(entry, shareScaleInput.checked);
      });
      shareScaleLabel.appendChild(shareScaleInput);
      shareScaleLabel.appendChild(document.createTextNode('Share colormap scale across fields'));
      optionGrid.appendChild(shareScaleLabel);
    }

    controlsBody.appendChild(optionGrid);

    const fieldNames = entry.layer.field_names || [];
    if (fieldNames.length > 1) {
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
      refreshFieldControls = () => updateFieldControls(Math.max(0, Math.min(fieldNames.length - 1, entry.activeFieldIdx ?? 0)));

      const selectField = (fieldIdx) => {
        const nextFieldIdx = Math.max(0, Math.min(fieldNames.length - 1, fieldIdx));
        setLayerField(entry, nextFieldIdx);
        updateFieldControls(nextFieldIdx);
      };

      fieldNames.forEach((name, fieldIdx) => {
        const item = document.createElement('div');
        item.className = 'custom-dropdown-item';
        if (fieldIdx === (entry.activeFieldIdx ?? 0)) item.classList.add('selected');
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
        closeDropdownMenu(menu);
        selectField((entry.activeFieldIdx ?? 0) - 1);
      });
      button.addEventListener('click', (ev) => {
        ev.stopPropagation();
        toggleDropdownMenu(button, menu);
      });
      nextButton.addEventListener('click', (ev) => {
        ev.stopPropagation();
        closeDropdownMenu(menu);
        selectField((entry.activeFieldIdx ?? 0) + 1);
      });

      updateFieldControls(Math.max(0, Math.min(fieldNames.length - 1, entry.activeFieldIdx ?? 0)));
      controls.appendChild(prevButton);
      dropdown.appendChild(button);
      dropdown.appendChild(menu);
      controls.appendChild(dropdown);
      controls.appendChild(nextButton);
      wrap.appendChild(controls);
      controlsBody.appendChild(wrap);
    }

    if (hasScalarOptions) {
      const colorbarHost = document.createElement('div');
      colorbarHost.className = 'layer-colorbar-host';
      controlsBody.appendChild(colorbarHost);
      entry.fieldOptionsContainer = colorbarHost;
    } else {
      entry.fieldOptionsContainer = null;
    }

    row.appendChild(checkbox);
    row.appendChild(info);
    if (toggleButton) row.appendChild(toggleButton);
    if (hasFieldOptions) row.appendChild(controlsBody);
    container.appendChild(row);
  });

  updateColorbars();
  updateInfoPanel();
  updateToggleAllLayers();
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

function setAllLayersVisibility(visible) {
  for (const entry of layers) {
    entry.visible = visible;
    if (entry.mesh) entry.mesh.visible = visible;
    if (entry.points) entry.points.visible = visible;
    if (entry.pointLines) entry.pointLines.visible = visible;
    updateEntryEdgeOverlay(entry);
  }
  rebuildLayerList();
  hideHoverTooltip();
  requestRender();
}

document.addEventListener('click', (ev) => {
  if (!ev.target.closest('.custom-dropdown') && !ev.target.closest('.custom-dropdown-menu')) {
    closeAllDropdownMenus();
  }
});

payload.layers.forEach((rawLayer) => {
  const layer = normalizeLayerData(rawLayer);
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
  const fieldCmaps = fieldCount > 0
    ? Array.from({ length: fieldCount }, (_value, idx) => ((fieldMetas && fieldMetas[idx] && fieldMetas[idx].cmap) ? fieldMetas[idx].cmap : NO_CMAP))
    : null;
  const entry = {
    layer,
    layerKind,
    mesh: layerKind === 'mesh' ? built.mesh : null,
    edgeLines: layerKind === 'mesh' ? built.edgeLines : null,
    points: layerKind === 'pointcloud' ? built.points : null,
    pointLines: layerKind === 'pointcloud' ? built.lines : null,
    mainObject: layerKind === 'mesh' ? built.mesh : built.points,
    colorGeometry: layerKind === 'mesh' ? built.mesh.geometry : built.points.geometry,
    visible: layer.visible !== false,
    activeFieldIdx: Number.isInteger(layer.active_field_idx) ? layer.active_field_idx : 0,
    activeValues: layer.values || null,
    activeCmap: layer.scalar_meta && layer.scalar_meta.cmap ? layer.scalar_meta.cmap : NO_CMAP,
    fieldCmaps,
    fieldClipStates: Array.from({ length: Math.max(fieldCount, 1) }, () => createClipState()),
    sharedFieldClip: createClipState(),
    shareFieldScale: Boolean(layer.share_field_scale ?? payload.initial_share_field_scale),
    opacity: Math.min(1.0, Math.max(0.05, Number(layer.opacity ?? payload.opacity) || 1.0)),
    showEdges: layerKind === 'mesh' && layer.show_edges === true,
    edgeThickness: getLayerEdgeThickness(layer),
    pointSize: Math.min(12.0, Math.max(1.0, Number(layer.point_size ?? payload.point_size) || 5.0)),
    optionsExpanded: layer.visible !== false,
    fieldOptionsContainer: null,
  };
  if (entry.mesh) entry.mesh.visible = entry.visible;
  if (entry.points) entry.points.visible = entry.visible;
  if (entry.pointLines) entry.pointLines.visible = entry.visible;
  updateEntryOpacity(entry, entry.opacity);
  updateEntryEdgeThickness(entry, entry.edgeThickness);
  if (entry.points) {
    entry.points.material.uniforms.size.value = entry.pointSize;
  }
  updateEntryEdgeOverlay(entry);
  layers.push(entry);

  const fieldNames = layer.field_names || null;
  if (fieldNames && fieldNames.length > 0) {
    setLayerField(entry, Math.max(0, Math.min(entry.activeFieldIdx, fieldNames.length - 1)));
  } else if (fieldValues || layer.values) {
    renderEntryColors(entry, entry.activeFieldIdx);
  }
});

const toggleAllLayersInput = document.getElementById('toggle-all-layers');
if (toggleAllLayersInput) {
  toggleAllLayersInput.addEventListener('change', () => {
    setAllLayersVisibility(toggleAllLayersInput.checked);
  });
}

rebuildLayerList();
refreshEdgeOverlay();

document.getElementById('btn-theme-mode').addEventListener('click', () => {
  applyThemeMode(!isDayMode);
});
const commandsToggleBtn = document.getElementById('btn-toggle-commands');
const controlsPanel = document.getElementById('controls');
if (commandsToggleBtn && controlsPanel) {
  const updateCommandsToggleButton = () => {
    commandsToggleBtn.textContent = commandsVisible ? 'Hide Commands' : 'Show Commands';
    commandsToggleBtn.classList.toggle('active', !commandsVisible);
  };
  commandsToggleBtn.addEventListener('click', () => {
    commandsVisible = !commandsVisible;
    controlsPanel.classList.toggle('collapsed', !commandsVisible);
    updateCommandsToggleButton();
  });
  updateCommandsToggleButton();
  controlsPanel.addEventListener('scroll', () => {
    closeAllDropdownMenus();
  }, { passive: true });
}
document.getElementById('btn-reset-camera').addEventListener('click', () => {
  fitCamera();
});
controls.addEventListener('change', () => {
  requestRender();
  scheduleHoverRefresh();
});
renderer.domElement.addEventListener('pointermove', (event) => {
  pointerState.inside = true;
  pointerState.clientX = event.clientX;
  pointerState.clientY = event.clientY;
  scheduleHoverRefresh();
});
renderer.domElement.addEventListener('pointerleave', () => {
  pointerState.inside = false;
  hideHoverTooltip();
});

document.getElementById('btn-quit-top').addEventListener('click', async () => {
  try {
    await fetch('/_shutdown', { method: 'POST' });
  } catch (_err) {
  }
  window.close();
  window.location.href = 'about:blank';
});

window.addEventListener('resize', () => {
  closeAllDropdownMenus();
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  refreshLineMaterialResolutions();
  controls.handleResize?.();
  requestRenderAndHover();
});

applyThemeMode(true);
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
if (layers.length > 0) {
  requestAnimationFrame(() => {
    fitCamera();
    renderer.render(scene, camera);
  });
}
animate();
