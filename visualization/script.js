/* ═══════════════════════════════════════════════════════════════
   PackOptima — Main Script
   Three.js 3D viewport + API integration + UI logic
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = "";  // same origin

// ── Color palette for boxes ────────────────────────────────────
const BOX_COLORS = [
    0x06b6d4, 0x8b5cf6, 0xec4899, 0x10b981, 0xf59e0b,
    0xef4444, 0x3b82f6, 0x14b8a6, 0xf97316, 0xa855f7,
    0x22d3ee, 0x84cc16, 0xe879f9, 0xfbbf24, 0x2dd4bf,
    0x60a5fa, 0xfb7185, 0x34d399, 0xa78bfa, 0xfb923c,
];

function getBoxColor(index) {
    return BOX_COLORS[index % BOX_COLORS.length];
}

function hexToCSS(hex) {
    const r = (hex >> 16) & 255;
    const g = (hex >> 8) & 255;
    const b = hex & 255;
    return `rgb(${r}, ${g}, ${b})`;
}

// ── State ──────────────────────────────────────────────────────
let boxes = [];
let currentPositions = [];
let scene, camera, renderer, controls;
let containerMesh = null;
let boxMeshes = [];
let animFrame;

// ── DOM References ─────────────────────────────────────────────
const containerL = document.getElementById("container-l");
const containerW = document.getElementById("container-w");
const containerH = document.getElementById("container-h");
const containerVolumeDisplay = document.getElementById("container-volume-display");
const boxTableBody = document.getElementById("box-table-body");
const boxCountBadge = document.getElementById("box-count-badge");
const emptyState = document.getElementById("empty-state");
const boxTableEl = document.getElementById("box-table");
const algorithmSelect = document.getElementById("algorithm-select");
const loadingOverlay = document.getElementById("loading-overlay");
const fpsCounter = document.getElementById("fps-counter");

// Result elements
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContent = document.getElementById("results-content");
const statUtilization = document.getElementById("stat-utilization");
const progressUtilization = document.getElementById("progress-utilization");
const statPlaced = document.getElementById("stat-placed");
const statTime = document.getElementById("stat-time");
const statFitness = document.getElementById("stat-fitness");
const statAlgorithm = document.getElementById("stat-algorithm");

// Comparison elements
const comparisonPlaceholder = document.getElementById("comparison-placeholder");
const comparisonContent = document.getElementById("comparison-content");
const comparisonTableBody = document.getElementById("comparison-table-body");
const comparisonChart = document.getElementById("comparison-chart");

// Legend
const boxLegend = document.getElementById("box-legend");


// ═══════════════════════════════════════════════════════════════
//  THREE.JS SETUP
// ═══════════════════════════════════════════════════════════════

function initThree() {
    const wrapper = document.getElementById("three-canvas-wrapper");
    const canvas = document.getElementById("three-canvas");

    scene = new THREE.Scene();

    // Camera
    camera = new THREE.PerspectiveCamera(
        45,
        wrapper.clientWidth / wrapper.clientHeight,
        0.1,
        50000
    );
    camera.position.set(800, 600, 1000);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true,
    });
    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0b0e14, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 100;
    controls.maxDistance = 5000;
    controls.target.set(295, 120, 118);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(500, 800, 400);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x8b5cf6, 0.2);
    fillLight.position.set(-300, 200, -200);
    scene.add(fillLight);

    // Grid helper on the floor
    const grid = new THREE.GridHelper(2000, 40, 0x1a1f2e, 0x111620);
    grid.position.y = -0.5;
    scene.add(grid);

    // Draw initial container
    drawContainer();

    // Resize handler
    window.addEventListener("resize", () => {
        camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
    });

    // FPS counter
    let frameCount = 0;
    let lastTime = performance.now();

    function animate() {
        animFrame = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);

        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
            fpsCounter.textContent = `${frameCount} FPS`;
            frameCount = 0;
            lastTime = now;
        }
    }
    animate();
}

function drawContainer() {
    // Remove old container mesh
    if (containerMesh) {
        scene.remove(containerMesh);
        containerMesh = null;
    }

    const L = parseFloat(containerL.value) || 590;
    const W = parseFloat(containerW.value) || 235;
    const H = parseFloat(containerH.value) || 239;

    // Wireframe box
    const geo = new THREE.BoxGeometry(L, H, W);
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({
        color: 0x2a3040,
        linewidth: 1,
    });
    containerMesh = new THREE.LineSegments(edges, lineMat);
    containerMesh.position.set(L / 2, H / 2, W / 2);
    scene.add(containerMesh);

    // Semi-transparent floor
    const floorGeo = new THREE.PlaneGeometry(L, W);
    const floorMat = new THREE.MeshStandardMaterial({
        color: 0x111620,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(L / 2, 0.01, W / 2);
    floor.receiveShadow = true;
    containerMesh.add(floor);

    // Update controls target
    controls.target.set(L / 2, H / 2, W / 2);
    camera.position.set(L * 1.3, H * 1.5, W * 2);
    controls.update();
}

// ── Animation state ────────────────────────────────────────────
let fillAnimationId = null;   // tracks any running fill animation
let fillAnimations = [];      // per-box tweens in progress

function clearBoxMeshes() {
    // Cancel any running fill animation
    if (fillAnimationId) {
        cancelAnimationFrame(fillAnimationId);
        fillAnimationId = null;
    }
    fillAnimations = [];
    boxMeshes.forEach(m => scene.remove(m));
    boxMeshes = [];
}

/**
 * Create a single box mesh + edge pair, add to scene, return { mesh, edge }.
 * Starts invisible (scale 0) so we can animate it in.
 */
function createBoxMesh(pos, colorIndex) {
    const color = getBoxColor(colorIndex);

    const geo = new THREE.BoxGeometry(pos.l, pos.h, pos.w);
    const mat = new THREE.MeshStandardMaterial({
        color: color,
        transparent: true,
        opacity: 0.0,       // start invisible
        roughness: 0.3,
        metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geo, mat);

    // Final resting position
    const targetX = pos.x + pos.l / 2;
    const targetY = pos.z + pos.h / 2;   // z in data => y in Three.js
    const targetZ = pos.y + pos.w / 2;

    // Start above the container and at zero scale
    mesh.position.set(targetX, targetY + 80, targetZ);
    mesh.scale.set(0.01, 0.01, 0.01);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    boxMeshes.push(mesh);

    // Edges
    const edgeGeo = new THREE.EdgesGeometry(geo);
    const edgeMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.0,
    });
    const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeMesh.position.copy(mesh.position);
    edgeMesh.scale.copy(mesh.scale);
    scene.add(edgeMesh);
    boxMeshes.push(edgeMesh);

    return { mesh, edgeMesh, targetX, targetY, targetZ };
}

/**
 * Animate boxes filling the container one-by-one.
 * Returns a Promise that resolves when the animation completes.
 */
function drawPackedBoxes(positions, animated = true) {
    clearBoxMeshes();

    if (!animated || positions.length === 0) {
        // Instant draw (no animation)
        positions.forEach((pos, i) => {
            const { mesh, edgeMesh, targetX, targetY, targetZ } = createBoxMesh(pos, i);
            mesh.position.set(targetX, targetY, targetZ);
            mesh.scale.set(1, 1, 1);
            mesh.material.opacity = 0.7;
            edgeMesh.position.set(targetX, targetY, targetZ);
            edgeMesh.scale.set(1, 1, 1);
            edgeMesh.material.opacity = 0.3;
        });
        return Promise.resolve();
    }

    // ── Animated fill ──────────────────────────────────────────
    return new Promise((resolve) => {
        // Compute adaptive delay: faster for large sets, slower for small
        const count = positions.length;
        const delayPerBox = count > 60 ? 40 : count > 30 ? 70 : count > 15 ? 100 : 150;
        const animDuration = 320; // ms per box tween

        let nextBoxIndex = 0;
        let lastSpawnTime = 0;

        // Show progress counter
        const counterEl = document.getElementById("fill-counter");
        if (counterEl) {
            counterEl.classList.remove("hidden");
            counterEl.textContent = `0 / ${count}`;
        }

        // Easing: ease-out cubic
        function easeOutCubic(t) {
            return 1 - Math.pow(1 - t, 3);
        }

        function tick(timestamp) {
            if (!lastSpawnTime) lastSpawnTime = timestamp;

            // Spawn new boxes at intervals
            while (
                nextBoxIndex < count &&
                timestamp - lastSpawnTime >= delayPerBox * nextBoxIndex
            ) {
                const pos = positions[nextBoxIndex];
                const idx = nextBoxIndex;
                const { mesh, edgeMesh, targetX, targetY, targetZ } = createBoxMesh(pos, idx);

                fillAnimations.push({
                    mesh, edgeMesh,
                    targetX, targetY, targetZ,
                    startTime: timestamp,
                    duration: animDuration,
                    done: false,
                });

                nextBoxIndex++;

                // Update counter
                if (counterEl) {
                    counterEl.textContent = `${nextBoxIndex} / ${count}`;
                }
            }

            // Update running tweens
            let allDone = nextBoxIndex >= count;
            for (const anim of fillAnimations) {
                if (anim.done) continue;

                const elapsed = timestamp - anim.startTime;
                let t = Math.min(elapsed / anim.duration, 1);
                const eased = easeOutCubic(t);

                // Scale: 0.01 → 1
                const s = 0.01 + eased * 0.99;
                anim.mesh.scale.set(s, s, s);
                anim.edgeMesh.scale.set(s, s, s);

                // Position: drop from +80 to target
                const dropOffset = (1 - eased) * 80;
                anim.mesh.position.set(anim.targetX, anim.targetY + dropOffset, anim.targetZ);
                anim.edgeMesh.position.set(anim.targetX, anim.targetY + dropOffset, anim.targetZ);

                // Opacity fade in
                anim.mesh.material.opacity = eased * 0.7;
                anim.edgeMesh.material.opacity = eased * 0.3;

                if (t >= 1) {
                    anim.done = true;
                } else {
                    allDone = false;
                }
            }

            if (!allDone || fillAnimations.some(a => !a.done)) {
                fillAnimationId = requestAnimationFrame(tick);
            } else {
                fillAnimationId = null;
                if (counterEl) counterEl.classList.add("hidden");
                resolve();
            }
        }

        fillAnimationId = requestAnimationFrame(tick);
    });
}


// ═══════════════════════════════════════════════════════════════
//  BOX TABLE MANAGEMENT
// ═══════════════════════════════════════════════════════════════

function updateBoxCount() {
    boxCountBadge.textContent = boxes.length;
    emptyState.classList.toggle("hidden", boxes.length > 0);
    boxTableEl.classList.toggle("hidden", boxes.length === 0);
}

function renderBoxTable() {
    boxTableBody.innerHTML = "";
    boxes.forEach((box, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${box.id}</td>
            <td><input type="number" value="${box.length}" data-idx="${i}" data-field="length" min="1" /></td>
            <td><input type="number" value="${box.width}" data-idx="${i}" data-field="width" min="1" /></td>
            <td><input type="number" value="${box.height}" data-idx="${i}" data-field="height" min="1" /></td>
            <td><input type="number" value="${box.weight}" data-idx="${i}" data-field="weight" min="0.1" step="0.1" /></td>
            <td>
                <button class="btn-remove-box" data-idx="${i}" title="Remove box">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </td>
        `;
        boxTableBody.appendChild(tr);
    });
    updateBoxCount();
}

function addBox(l = 50, w = 40, h = 30, weight = 10) {
    const id = boxes.length > 0 ? Math.max(...boxes.map(b => b.id)) + 1 : 1;
    boxes.push({ id, length: l, width: w, height: h, weight });
    renderBoxTable();
}

function removeBox(idx) {
    boxes.splice(idx, 1);
    // Re-number IDs
    boxes.forEach((b, i) => b.id = i + 1);
    renderBoxTable();
}

function clearBoxes() {
    boxes = [];
    renderBoxTable();
}

// Table change handler (event delegation)
boxTableBody.addEventListener("input", (e) => {
    if (e.target.tagName === "INPUT") {
        const idx = parseInt(e.target.dataset.idx);
        const field = e.target.dataset.field;
        const val = parseFloat(e.target.value) || 0;
        if (boxes[idx]) {
            boxes[idx][field] = val;
        }
    }
});

boxTableBody.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-remove-box");
    if (btn) {
        removeBox(parseInt(btn.dataset.idx));
    }
});


// ═══════════════════════════════════════════════════════════════
//  CONTAINER VOLUME DISPLAY
// ═══════════════════════════════════════════════════════════════

function updateContainerVolume() {
    const L = parseFloat(containerL.value) || 0;
    const W = parseFloat(containerW.value) || 0;
    const H = parseFloat(containerH.value) || 0;
    const vol = L * W * H;
    containerVolumeDisplay.textContent = `Volume: ${vol.toLocaleString()} cm³`;
    drawContainer();
}

containerL.addEventListener("input", updateContainerVolume);
containerW.addEventListener("input", updateContainerVolume);
containerH.addEventListener("input", updateContainerVolume);


// ═══════════════════════════════════════════════════════════════
//  API CALLS
// ═══════════════════════════════════════════════════════════════

function getContainerData() {
    return {
        length: parseFloat(containerL.value) || 590,
        width: parseFloat(containerW.value) || 235,
        height: parseFloat(containerH.value) || 239,
    };
}

function showLoading(msg = "Running algorithm…") {
    loadingOverlay.querySelector(".loading-text").textContent = msg;
    loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
    loadingOverlay.classList.add("hidden");
}

async function loadDataset() {
    try {
        showLoading("Loading dataset…");
        const res = await fetch(`${API_BASE}/api/dataset`);
        const data = await res.json();

        // Set container
        containerL.value = data.container.length;
        containerW.value = data.container.width;
        containerH.value = data.container.height;
        updateContainerVolume();

        // Set boxes
        boxes = data.boxes.map(b => ({
            id: b.id,
            length: b.length,
            width: b.width,
            height: b.height,
            weight: b.weight,
        }));
        renderBoxTable();
        hideLoading();
    } catch (err) {
        hideLoading();
        alert("Failed to load dataset: " + err.message);
    }
}

async function runSolver() {
    if (boxes.length === 0) {
        alert("Please add boxes first!");
        return;
    }

    const algo = algorithmSelect.value;
    const algoNames = { ga: "Genetic Algorithm", greedy: "Greedy (BFD)", sa: "Simulated Annealing" };
    showLoading(`Running ${algoNames[algo] || algo}…`);

    try {
        const res = await fetch(`${API_BASE}/api/solve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                boxes: boxes,
                container: getContainerData(),
                algorithm: algo,
            }),
        });

        const result = await res.json();

        if (result.error) {
            hideLoading();
            alert("Error: " + result.error);
            return;
        }

        currentPositions = result.positions;
        displayResults(result);
        updateLegend(result.positions);
        hideLoading(); // hide spinner so user sees the 3D animation
        await drawPackedBoxes(result.positions);
    } catch (err) {
        hideLoading();
        alert("Solver failed: " + err.message);
    }
}

async function runCompare() {
    if (boxes.length === 0) {
        alert("Please add boxes first!");
        return;
    }

    showLoading("Running all algorithms…");

    try {
        const res = await fetch(`${API_BASE}/api/compare`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                boxes: boxes,
                container: getContainerData(),
            }),
        });

        const data = await res.json();

        if (data.error) {
            hideLoading();
            alert("Error: " + data.error);
            return;
        }

        displayComparison(data.results);

        // Show the best result in the right-panel results
        const best = data.results.reduce((a, b) =>
            a.utilization > b.utilization ? a : b
        );
        currentPositions = best.positions;
        displayResults(best);
        updateLegend(best.positions);
        hideLoading();

        // Show comparison multi-viewport
        showComparisonViewport(data.results);
    } catch (err) {
        hideLoading();
        alert("Comparison failed: " + err.message);
    }
}


// ═══════════════════════════════════════════════════════════════
//  COMPARISON MULTI-VIEWPORT (side-by-side 3D scenes)
// ═══════════════════════════════════════════════════════════════

const comparisonViewport = document.getElementById("comparison-viewport");
const threeCanvasWrapper = document.getElementById("three-canvas-wrapper");
let compareScenes = {};       // { ga: { scene, camera, renderer, controls, meshes }, ... }
let compareAnimFrameId = null;

function showComparisonViewport(results) {
    // Hide main single viewport, show comparison
    threeCanvasWrapper.style.display = "none";
    document.getElementById("viewport-watermark").style.display = "none";
    document.getElementById("viewport-overlay").style.display = "none";
    comparisonViewport.classList.remove("hidden");

    // Find best
    const bestUtil = Math.max(...results.map(r => r.utilization));

    // Init each cell
    results.forEach(r => {
        const algo = r.algorithm;
        const cell = document.getElementById(`compare-cell-${algo}`);
        const canvas = document.getElementById(`compare-canvas-${algo}`);
        const statEl = document.getElementById(`compare-stat-${algo}`);

        // Mark best
        cell.classList.toggle("best-cell", r.utilization === bestUtil);

        // Stat line
        statEl.textContent = `${r.utilization.toFixed(1)}%  ·  ${r.boxesPlaced}/${r.totalBoxes} boxes  ·  ${r.executionTime.toFixed(3)}s`;

        // Init Three.js scene for this cell
        initCompareScene(algo, canvas, r.positions);
    });

    // Start animation loop for all comparison scenes
    startCompareAnimation();
}

function hideComparisonViewport() {
    comparisonViewport.classList.add("hidden");
    threeCanvasWrapper.style.display = "";
    document.getElementById("viewport-watermark").style.display = "";
    document.getElementById("viewport-overlay").style.display = "";

    // Clean up comparison scenes
    stopCompareAnimation();
    for (const algo of Object.keys(compareScenes)) {
        const cs = compareScenes[algo];
        cs.renderer.dispose();
        if (cs.controls) cs.controls.dispose();
    }
    compareScenes = {};

    // Clear best-cell classes
    document.querySelectorAll(".compare-cell").forEach(c => c.classList.remove("best-cell"));
}

function initCompareScene(algo, canvas, positions) {
    const wrapper = canvas.parentElement;
    const w = wrapper.clientWidth || 400;
    const h = wrapper.clientHeight || 300;

    const cScene = new THREE.Scene();

    const cCamera = new THREE.PerspectiveCamera(45, w / h, 0.1, 50000);

    const cRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    cRenderer.setSize(w, h);
    cRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    cRenderer.setClearColor(0x0b0e14, 1);
    cRenderer.shadowMap.enabled = true;

    const cControls = new THREE.OrbitControls(cCamera, cRenderer.domElement);
    cControls.enableDamping = true;
    cControls.dampingFactor = 0.08;
    cControls.minDistance = 100;
    cControls.maxDistance = 5000;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    cScene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(500, 800, 400);
    dirLight.castShadow = true;
    cScene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x8b5cf6, 0.2);
    fillLight.position.set(-300, 200, -200);
    cScene.add(fillLight);

    // Grid
    const grid = new THREE.GridHelper(2000, 40, 0x1a1f2e, 0x111620);
    grid.position.y = -0.5;
    cScene.add(grid);

    // Container wireframe
    const L = parseFloat(containerL.value) || 590;
    const W = parseFloat(containerW.value) || 235;
    const H = parseFloat(containerH.value) || 239;

    const geo = new THREE.BoxGeometry(L, H, W);
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x2a3040 });
    const contMesh = new THREE.LineSegments(edges, lineMat);
    contMesh.position.set(L / 2, H / 2, W / 2);
    cScene.add(contMesh);

    // Floor
    const floorGeo = new THREE.PlaneGeometry(L, W);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x111620, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(L / 2, 0.01, W / 2);
    floor.receiveShadow = true;
    contMesh.add(floor);

    // Camera position
    cCamera.position.set(L * 1.3, H * 1.5, W * 2);
    cControls.target.set(L / 2, H / 2, W / 2);
    cControls.update();

    // Draw packed boxes (instant, no animation)
    const meshes = [];
    if (positions && positions.length > 0) {
        positions.forEach((pos, i) => {
            const color = getBoxColor(i);
            const boxGeo = new THREE.BoxGeometry(pos.l, pos.h, pos.w);
            const boxMat = new THREE.MeshStandardMaterial({
                color, transparent: true, opacity: 0.7, roughness: 0.3, metalness: 0.1,
            });
            const mesh = new THREE.Mesh(boxGeo, boxMat);
            mesh.position.set(pos.x + pos.l / 2, pos.z + pos.h / 2, pos.y + pos.w / 2);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            cScene.add(mesh);
            meshes.push(mesh);

            // Edges
            const edgeGeo = new THREE.EdgesGeometry(boxGeo);
            const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
            const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
            edgeMesh.position.copy(mesh.position);
            cScene.add(edgeMesh);
            meshes.push(edgeMesh);
        });
    }

    compareScenes[algo] = { scene: cScene, camera: cCamera, renderer: cRenderer, controls: cControls, meshes, wrapper };
}

function startCompareAnimation() {
    stopCompareAnimation();

    function tick() {
        for (const algo of Object.keys(compareScenes)) {
            const cs = compareScenes[algo];

            // Handle resize
            const w = cs.wrapper.clientWidth;
            const h = cs.wrapper.clientHeight;
            if (w > 0 && h > 0) {
                if (cs.renderer.domElement.width !== w || cs.renderer.domElement.height !== h) {
                    cs.renderer.setSize(w, h);
                    cs.camera.aspect = w / h;
                    cs.camera.updateProjectionMatrix();
                }
            }

            cs.controls.update();
            cs.renderer.render(cs.scene, cs.camera);
        }
        compareAnimFrameId = requestAnimationFrame(tick);
    }
    compareAnimFrameId = requestAnimationFrame(tick);
}

function stopCompareAnimation() {
    if (compareAnimFrameId) {
        cancelAnimationFrame(compareAnimFrameId);
        compareAnimFrameId = null;
    }
}


// ═══════════════════════════════════════════════════════════════
//  RESULTS DISPLAY
// ═══════════════════════════════════════════════════════════════

const ALGO_DISPLAY_NAMES = {
    ga: "🧬 Genetic Algorithm",
    greedy: "📦 Greedy (BFD)",
    sa: "🌡️ Simulated Annealing",
};

function displayResults(result) {
    resultsPlaceholder.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    const util = result.utilization;
    statUtilization.textContent = `${util.toFixed(1)}%`;
    progressUtilization.style.width = `${Math.min(util, 100)}%`;

    // Color the progress bar based on utilization
    if (util >= 80) {
        progressUtilization.style.background = "var(--gradient-success)";
    } else if (util >= 60) {
        progressUtilization.style.background = "var(--gradient-main)";
    } else {
        progressUtilization.style.background = "var(--gradient-warm)";
    }

    statPlaced.textContent = `${result.boxesPlaced} / ${result.totalBoxes}`;
    statTime.textContent = `${result.executionTime}s`;
    statFitness.textContent = result.fitness.toFixed(2);
    statAlgorithm.textContent = ALGO_DISPLAY_NAMES[result.algorithm] || result.algorithm;
}

function displayComparison(results) {
    comparisonPlaceholder.classList.add("hidden");
    comparisonContent.classList.remove("hidden");

    // Find best utilization
    const bestUtil = Math.max(...results.map(r => r.utilization));

    // Table
    comparisonTableBody.innerHTML = "";
    results.forEach(r => {
        const isBest = r.utilization === bestUtil;
        const tr = document.createElement("tr");
        if (isBest) tr.classList.add("best-row");
        tr.innerHTML = `
            <td>${ALGO_DISPLAY_NAMES[r.algorithm] || r.algorithm}${isBest ? ' ★' : ''}</td>
            <td>${r.utilization.toFixed(1)}%</td>
            <td>${r.boxesPlaced}/${r.totalBoxes}</td>
            <td>${r.executionTime.toFixed(3)}</td>
        `;
        comparisonTableBody.appendChild(tr);
    });

    // Bar chart (custom CSS-based)
    const chartWrapper = document.querySelector(".chart-wrapper");
    // Use the canvas area for a simple bar chart
    chartWrapper.innerHTML = "";

    const barColors = {
        ga: "var(--gradient-main)",
        greedy: "var(--gradient-success)",
        sa: "var(--gradient-warm)",
    };

    results.forEach(r => {
        const group = document.createElement("div");
        group.className = "chart-bar-group";

        const label = document.createElement("span");
        label.className = "chart-bar-label";
        label.textContent = r.algorithm.toUpperCase();

        const track = document.createElement("div");
        track.className = "chart-bar-track";

        const fill = document.createElement("div");
        fill.className = "chart-bar-fill";
        fill.style.background = barColors[r.algorithm] || "var(--gradient-main)";
        fill.style.width = "0%";

        const value = document.createElement("span");
        value.className = "chart-bar-value";
        value.textContent = `${r.utilization.toFixed(1)}%`;

        fill.appendChild(value);
        track.appendChild(fill);
        group.appendChild(label);
        group.appendChild(track);
        chartWrapper.appendChild(group);

        // Animate
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                fill.style.width = `${Math.min(r.utilization, 100)}%`;
            });
        });
    });
}

function updateLegend(positions) {
    boxLegend.innerHTML = "";
    if (!positions || positions.length === 0) {
        boxLegend.innerHTML = '<p class="empty-state">No boxes placed.</p>';
        return;
    }

    positions.forEach((pos, i) => {
        const chip = document.createElement("div");
        chip.className = "legend-chip";

        const colorDot = document.createElement("div");
        colorDot.className = "legend-color";
        colorDot.style.background = hexToCSS(getBoxColor(i));

        const text = document.createElement("span");
        text.textContent = `#${pos.id} ${pos.l}×${pos.w}×${pos.h}`;

        chip.appendChild(colorDot);
        chip.appendChild(text);
        boxLegend.appendChild(chip);
    });
}


// ═══════════════════════════════════════════════════════════════
//  EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════

document.getElementById("btn-load-dataset").addEventListener("click", loadDataset);
document.getElementById("btn-add-box").addEventListener("click", () => addBox());
document.getElementById("btn-clear-boxes").addEventListener("click", clearBoxes);
document.getElementById("btn-solve").addEventListener("click", runSolver);
document.getElementById("btn-compare").addEventListener("click", runCompare);
document.getElementById("btn-back-single").addEventListener("click", hideComparisonViewport);


// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════

updateBoxCount();
initThree();
updateContainerVolume();

