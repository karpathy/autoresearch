// ─── Main Application ─────────────────────────────────────────────────────────
(function () {
  'use strict';

  const C = {
    bg:       '#0a0a0f',
    bg2:      '#111118',
    bg3:      '#1a1a26',
    border:   '#2a2a3a',
    text:     '#e8e8f0',
    muted:    '#8888aa',
    accent:   '#7c6dfa',
    accent2:  '#fa6d6d',
    accent3:  '#6dfabc',
    wire:     '#2a2a42',
    wireHi:   '#7c6dfa',
    gate:     '#7c6dfa',
    gateSwap: '#fa6d6d',
  };

  // ── Hero canvas — ambient animated network ─────────────────────────────────
  (function initHero() {
    const canvas = document.getElementById('hero-canvas');
    const ctx    = canvas.getContext('2d');
    const N      = 16;
    const net    = NETWORKS.batcher;
    let W, H;
    const particles = [];

    class Particle {
      constructor() { this.reset(true); }
      reset(randomX) {
        this.wire  = Math.floor(Math.random() * N);
        this.x     = randomX ? Math.random() * (W || 800) : 80;
        this.speed = 0.5 + Math.random() * 1.2;
        this.alpha = 0.2 + Math.random() * 0.5;
        this.r     = 1.5 + Math.random() * 2;
      }
      update() { this.x += this.speed; if (this.x > (W || 800)) this.reset(false); }
      draw(margin, wireSpacing) {
        const y = margin + this.wire * wireSpacing;
        ctx.beginPath();
        ctx.arc(this.x, y, this.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(124,109,250,${this.alpha})`;
        ctx.fill();
      }
    }

    function resize() {
      W = canvas.width  = canvas.offsetWidth;
      H = canvas.height = canvas.offsetHeight;
    }

    function frame() {
      ctx.clearRect(0, 0, W, H);
      const margin      = Math.min(80, H * 0.08);
      const wireSpacing = (H - margin * 2) / (N - 1);
      const layerWidth  = (W - margin * 2) / (net.layers.length + 1);

      // Wires
      for (let i = 0; i < N; i++) {
        const y = margin + i * wireSpacing;
        ctx.beginPath();
        ctx.moveTo(margin, y); ctx.lineTo(W - margin, y);
        ctx.strokeStyle = '#1e1e30';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Comparators
      net.layers.forEach((layer, li) => {
        const x = margin + (li + 1) * layerWidth;
        layer.forEach(([a, b]) => {
          const y1 = margin + a * wireSpacing;
          const y2 = margin + b * wireSpacing;
          ctx.beginPath();
          ctx.moveTo(x, y1); ctx.lineTo(x, y2);
          ctx.strokeStyle = '#252538';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          [y1, y2].forEach(y => {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#2e2e50';
            ctx.fill();
          });
        });
      });

      // Particles
      particles.forEach(p => { p.update(); p.draw(margin, wireSpacing); });
      requestAnimationFrame(frame);
    }

    window.addEventListener('resize', resize);
    resize();
    for (let i = 0; i < 50; i++) particles.push(new Particle());
    frame();

    document.getElementById('explore-btn').addEventListener('click', () => {
      document.getElementById('interactive').scrollIntoView({ behavior: 'smooth' });
    });
  })();

  // ── Comparator demo ────────────────────────────────────────────────────────
  (function initComparatorDemo() {
    const canvas = document.getElementById('comparator-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    let t = 0, v1 = 7, v2 = 3;

    function frame() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.bg2;
      ctx.fillRect(0, 0, W, H);

      const y1 = H * 0.33, y2 = H * 0.67;
      const xL = 40, xMid = W / 2, xR = W - 40;

      // Wires
      [y1, y2].forEach(y => {
        ctx.beginPath();
        ctx.moveTo(xL, y); ctx.lineTo(xR, y);
        ctx.strokeStyle = C.wire; ctx.lineWidth = 1.5; ctx.stroke();
      });

      // Gate
      ctx.beginPath();
      ctx.moveTo(xMid, y1); ctx.lineTo(xMid, y2);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.stroke();
      [y1, y2].forEach(y => {
        ctx.beginPath(); ctx.arc(xMid, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = C.accent; ctx.fill();
      });

      const phase   = (t % 200) / 200;
      const showing = phase > 0.35 && phase < 0.85;
      const didSwap = v1 > v2;

      ctx.font      = 'bold 15px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';

      // Left
      ctx.fillStyle = C.muted;
      ctx.fillText(v1, xL + 28, y1 + 5);
      ctx.fillText(v2, xL + 28, y2 + 5);

      // Right (result)
      const out1 = (showing && didSwap) ? v2 : v1;
      const out2 = (showing && didSwap) ? v1 : v2;
      ctx.fillStyle = (showing && didSwap) ? C.accent2 : C.accent3;
      ctx.fillText(out1, xR - 28, y1 + 5);
      ctx.fillStyle = (showing && didSwap) ? C.accent3 : C.accent2;
      ctx.fillText(out2, xR - 28, y2 + 5);

      // Swap arrow
      if (showing && didSwap) {
        const alpha = Math.sin((phase - 0.35) / 0.5 * Math.PI);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle   = C.accent2;
        ctx.font        = '20px sans-serif';
        ctx.fillText('⇅', xMid + 22, (y1 + y2) / 2 + 7);
        ctx.restore();
      }

      // Label
      if (showing && didSwap) {
        ctx.fillStyle = C.accent2;
        ctx.font      = 'bold 10px Inter, system-ui';
        ctx.fillText('SWAP', xMid, y2 + 26);
      } else if (showing) {
        ctx.fillStyle = C.accent3;
        ctx.font      = 'bold 10px Inter, system-ui';
        ctx.fillText('in order', xMid, y2 + 26);
      }

      t++;
      if (t % 250 === 0) {
        v1 = Math.floor(Math.random() * 9) + 1;
        v2 = Math.floor(Math.random() * 9) + 1;
        while (v1 === v2) v2 = Math.floor(Math.random() * 9) + 1;
      }
      requestAnimationFrame(frame);
    }
    frame();
  })();

  // ── Interactive Network Visualizer ────────────────────────────────────────
  (function initNetworkViz() {
    const canvas     = document.getElementById('network-canvas');
    const ctx        = canvas.getContext('2d');
    const inputLane  = document.getElementById('input-lane');
    const outputLane = document.getElementById('output-lane');
    const stepCounterEl = document.getElementById('current-step');
    const totalStepsEl  = document.getElementById('total-steps');
    const flashEl       = document.getElementById('comparison-flash');

    let currentNetKey = 'batcher';
    let values    = [];
    let stepIndex = 0;
    let playing   = false;
    let playTimer = null;
    let speed     = 5;
    let allSteps  = [];
    let W = 0, H = 0;
    let animReq   = null;
    let highlightFade = 0; // 0-1, for glow animation

    function getNet() { return NETWORKS[currentNetKey]; }

    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    }

    function initValues() {
      const n = getNet().n;
      values = Array.from({ length: n }, (_, i) => i + 1);
      shuffle(values);
    }

    function buildSteps() {
      const net = getNet();
      const arr = [...values];
      allSteps  = [{ layerIdx: -1, comparator: null, arr: [...arr] }];

      net.layers.forEach((layer, li) => {
        layer.forEach(([i, j]) => {
          const swapped = arr[i] > arr[j];
          if (swapped) [arr[i], arr[j]] = [arr[j], arr[i]];
          allSteps.push({ layerIdx: li, comparator: [i, j], arr: [...arr], swapped });
        });
      });

      totalStepsEl.textContent = allSteps.length - 1;
    }

    // ── Layout helpers ────────────────────────────────────────────────────
    function resize() {
      const net  = getNet();
      const dpr  = devicePixelRatio;
      W = canvas.offsetWidth;
      H = Math.max(280, net.n * 32 + 80);
      canvas.width  = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function wireY(i) {
      const net    = getNet();
      const marginV = 36;
      return marginV + i * (H - marginV * 2) / Math.max(net.n - 1, 1);
    }

    function layerX(li) {
      const net    = getNet();
      const marginH = 72;
      return marginH + (li + 0.5) * (W - marginH * 2) / net.layers.length;
    }

    // ── Drawing ───────────────────────────────────────────────────────────
    function draw() {
      ctx.clearRect(0, 0, W, H);

      const net   = getNet();
      const step  = allSteps[stepIndex];
      const curLi = step ? step.layerIdx : -1;
      const curC  = step ? step.comparator : null;

      // Background gradient suggestion
      const bgGrad = ctx.createLinearGradient(0, 0, W, H);
      bgGrad.addColorStop(0, '#0d0d18');
      bgGrad.addColorStop(1, '#0a0a14');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // Wires
      for (let i = 0; i < net.n; i++) {
        const y = wireY(i);
        ctx.beginPath();
        ctx.moveTo(60, y); ctx.lineTo(W - 60, y);
        ctx.strokeStyle = C.wire;
        ctx.lineWidth   = 1.2;
        ctx.stroke();
      }

      // Layer depth tick marks at top
      net.layers.forEach((_, li) => {
        const x = layerX(li);
        const active = li === curLi;
        ctx.fillStyle = active ? C.accent + 'cc' : C.muted + '44';
        ctx.font      = `${active ? 'bold ' : ''}8px Inter, system-ui`;
        ctx.textAlign = 'center';
        ctx.fillText(li + 1, x, 12);
      });

      // Comparators
      net.layers.forEach((layer, li) => {
        const x       = layerX(li);
        const isPast  = li < curLi;
        const isCur   = li === curLi;

        layer.forEach(([a, b]) => {
          const isThis = isCur && curC && curC[0] === a && curC[1] === b;
          const y1 = wireY(a), y2 = wireY(b);

          let barColor, dotColor, lineW;
          if (isThis) {
            const col  = step.swapped ? C.gateSwap : C.accent3;
            barColor   = col;
            dotColor   = col;
            lineW      = 2.5;
          } else if (isCur) {
            barColor   = C.accent + '88';
            dotColor   = C.accent + '88';
            lineW      = 1.8;
          } else if (isPast) {
            barColor   = C.accent + '55';
            dotColor   = C.accent + '55';
            lineW      = 1.2;
          } else {
            barColor   = C.border;
            dotColor   = C.accent + '33';
            lineW      = 1.2;
          }

          // Glow (active comparator)
          if (isThis) {
            const col = step.swapped ? C.gateSwap : C.accent3;
            const r   = Math.abs(y2 - y1) / 2 + 8;
            const grad = ctx.createRadialGradient(x, (y1+y2)/2, 0, x, (y1+y2)/2, r);
            grad.addColorStop(0, col + '40');
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(x, (y1+y2)/2, r, 0, Math.PI*2);
            ctx.fill();
          }

          // Bar
          ctx.beginPath();
          ctx.moveTo(x, y1); ctx.lineTo(x, y2);
          ctx.strokeStyle = barColor;
          ctx.lineWidth   = lineW;
          ctx.stroke();

          // Endpoints
          const dr = isThis ? 5 : 3.5;
          [y1, y2].forEach(y => {
            ctx.beginPath();
            ctx.arc(x, y, dr, 0, Math.PI * 2);
            ctx.fillStyle = dotColor;
            ctx.fill();
          });
        });
      });

      // Value indicators on right side
      if (step && step.arr) {
        const arr    = step.arr;
        const n      = net.n;
        const isFin  = stepIndex === allSteps.length - 1;

        for (let i = 0; i < n; i++) {
          const y   = wireY(i);
          const val = arr[i];
          const t   = (val - 1) / (n - 1);
          // Purple → green gradient by value
          const r   = Math.round(124 * (1-t) + 109 * t);
          const g   = Math.round(109 * (1-t) + 250 * t);
          const bl  = Math.round(250 * (1-t) + 188 * t);

          ctx.beginPath();
          ctx.arc(W - 62, y, isFin ? 6 : 4.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgb(${r},${g},${bl})`;
          ctx.fill();

          if (isFin) {
            ctx.fillStyle = '#fff8';
            ctx.font      = 'bold 8px Inter, system-ui';
            ctx.textAlign = 'center';
            ctx.fillText(val, W - 62, y + 3);
          }
        }
      }

      updateChips();
    }

    // ── Chips ─────────────────────────────────────────────────────────────
    function buildChips() {
      inputLane.innerHTML  = '';
      outputLane.innerHTML = '';
      const net     = getNet();
      const initArr = allSteps[0] ? allSteps[0].arr : values;

      for (let i = 0; i < net.n; i++) {
        const chip = makeChip(initArr[i]);
        chip.draggable   = true;
        chip.dataset.idx = i;
        chip.addEventListener('dragstart', e => { e.dataTransfer.setData('idx', i); chip.classList.add('dragging'); });
        chip.addEventListener('dragend',   () => chip.classList.remove('dragging'));
        chip.addEventListener('dragover',  e => e.preventDefault());
        chip.addEventListener('drop', e => {
          e.preventDefault();
          const from = parseInt(e.dataTransfer.getData('idx'));
          if (from !== i) { [values[from], values[i]] = [values[i], values[from]]; reset(); }
        });
        inputLane.appendChild(chip);
      }

      for (let i = 0; i < net.n; i++) {
        outputLane.appendChild(makeChip('·'));
      }
    }

    function makeChip(val) {
      const c = document.createElement('div');
      c.className   = 'chip';
      c.textContent = val;
      return c;
    }

    function updateChips() {
      const step  = allSteps[stepIndex];
      if (!step) return;
      const arr     = step.arr;
      const n       = getNet().n;
      const isFinal = stepIndex === allSteps.length - 1;
      const init    = allSteps[0].arr;

      inputLane.querySelectorAll('.chip').forEach((c, i)  => { c.textContent = init[i]; c.className = 'chip'; });
      outputLane.querySelectorAll('.chip').forEach((c, i) => {
        c.textContent = stepIndex === 0 ? '·' : arr[i];
        c.className   = isFinal ? 'chip sorted' : 'chip';
      });
      stepCounterEl.textContent = stepIndex;
    }

    // ── Flash ─────────────────────────────────────────────────────────────
    function showFlash(layerIdx, a, b, valA, valB, swapped) {
      const verdict = swapped ? 'SWAP' : 'keep';
      const cls     = swapped ? 'swap' : 'keep';
      const op      = swapped ? '>' : '≤';
      flashEl.innerHTML =
        `Layer ${layerIdx + 1}: wires ${a+1} &amp; ${b+1} &mdash; ${valA} ${op} ${valB},` +
        ` <span class="verdict ${cls}">${verdict}</span>`;
    }

    function clearFlash() {
      flashEl.innerHTML = '';
    }

    // ── Playback ──────────────────────────────────────────────────────────
    function stepForward() {
      if (stepIndex >= allSteps.length - 1) { stopPlay(); return; }
      stepIndex++;
      const s = allSteps[stepIndex];
      if (s.comparator) {
        const [a, b] = s.comparator;
        const prev   = allSteps[stepIndex - 1].arr;
        showFlash(s.layerIdx, a, b, prev[a], prev[b], s.swapped);
      } else {
        clearFlash();
      }
      draw();
      if (stepIndex === allSteps.length - 1) stopPlay();
    }

    function stepBack() {
      if (stepIndex > 0) {
        stepIndex--;
        const s = allSteps[stepIndex];
        if (s && s.comparator) {
          const [a, b] = s.comparator;
          const prev = allSteps[stepIndex - 1] ? allSteps[stepIndex - 1].arr : allSteps[0].arr;
          showFlash(s.layerIdx, a, b, prev[a], prev[b], s.swapped);
        } else {
          clearFlash();
        }
        draw();
      }
    }

    function startPlay() {
      if (playing) return;
      playing = true;
      document.getElementById('btn-play').textContent = '⏸ Pause';
      const ms = Math.round(900 / (speed * 1.8));
      playTimer = setInterval(stepForward, ms);
    }

    function stopPlay() {
      playing = false;
      clearInterval(playTimer);
      document.getElementById('btn-play').textContent = '▶ Play';
    }

    function reset() {
      stopPlay();
      buildSteps();
      stepIndex = 0;
      clearFlash();
      buildChips();
      resize();
      draw();
    }

    // ── Controls ──────────────────────────────────────────────────────────
    document.getElementById('btn-play').addEventListener('click', () => {
      if (playing) { stopPlay(); }
      else { if (stepIndex >= allSteps.length - 1) stepIndex = 0; startPlay(); }
    });
    document.getElementById('btn-step-fwd').addEventListener('click',  () => { stopPlay(); stepForward(); });
    document.getElementById('btn-step-back').addEventListener('click', () => { stopPlay(); stepBack(); });
    document.getElementById('btn-reset').addEventListener('click', reset);
    document.getElementById('btn-randomize').addEventListener('click', () => { shuffle(values); reset(); });

    document.getElementById('speed-slider').addEventListener('input', e => {
      speed = parseInt(e.target.value);
      if (playing) { stopPlay(); startPlay(); }
    });

    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentNetKey = tab.dataset.network;
        initValues();
        reset();
      });
    });

    window.addEventListener('resize', () => { resize(); draw(); });

    // ── Init ──────────────────────────────────────────────────────────────
    initValues();
    resize();
    buildSteps();
    buildChips();
    draw();
  })();

  // ── Compare Networks Chart ─────────────────────────────────────────────────
  (function initCompare() {
    const canvas = document.getElementById('compare-canvas');
    const ctx    = canvas.getContext('2d');

    const data = [
      { label: 'n=4',          comparators: 5,   depth: 3,  optimal: true  },
      { label: 'n=8',          comparators: 19,  depth: 6,  optimal: true  },
      { label: 'n=12',         comparators: 39,  depth: 9,  optimal: false },
      { label: 'n=16 lower\nbound', comparators: 53, depth: 10, optimal: false, isLower: true },
      { label: 'n=16 best\nknown',  comparators: 60, depth: 10, optimal: false, isBest: true  },
      { label: 'n=16\nBatcher', comparators: 63,  depth: 10, optimal: false },
      { label: 'n=32\nBatcher', comparators: 185, depth: 15, optimal: false },
    ];

    function draw() {
      const dpr = devicePixelRatio;
      const W   = canvas.offsetWidth;
      const H   = 300;
      canvas.width  = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.fillStyle = C.bg3;
      ctx.fillRect(0, 0, W, H);

      const padL = 55, padR = 20, padT = 40, padB = 60;
      const chartW = W - padL - padR;
      const chartH = H - padT - padB;
      const maxV   = Math.max(...data.map(d => d.comparators));
      const barW   = chartW / data.length;
      const barPad = barW * 0.18;

      // Grid
      [0, 0.25, 0.5, 0.75, 1].forEach(f => {
        const y = padT + chartH * (1 - f);
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + chartW, y);
        ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = C.muted; ctx.font = '9px Inter, system-ui';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(f * maxV), padL - 6, y + 3);
      });

      // Y label
      ctx.save();
      ctx.translate(13, padT + chartH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = C.muted; ctx.font = '10px Inter, system-ui'; ctx.textAlign = 'center';
      ctx.fillText('Comparators', 0, 0);
      ctx.restore();

      data.forEach((d, i) => {
        const bx  = padL + i * barW + barPad;
        const bw  = barW - barPad * 2;
        const bh  = (d.comparators / maxV) * chartH;
        const by  = padT + chartH - bh;

        // Bar color
        let topCol, botCol;
        if (d.optimal)  { topCol = C.accent3 + 'dd'; botCol = C.accent3 + '44'; }
        else if (d.isLower) { topCol = C.muted + 'cc'; botCol = C.muted + '33'; }
        else if (d.isBest)  { topCol = C.accent2 + 'cc'; botCol = C.accent2 + '33'; }
        else                { topCol = C.accent + 'cc'; botCol = C.accent + '33'; }

        const grad = ctx.createLinearGradient(0, by, 0, by + bh);
        grad.addColorStop(0, topCol); grad.addColorStop(1, botCol);
        ctx.fillStyle = grad;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, [3,3,0,0]);
        else { ctx.rect(bx, by, bw, bh); }
        ctx.fill();

        // Count label
        ctx.fillStyle = d.optimal ? C.accent3 : d.isLower ? C.muted : d.isBest ? C.accent2 : C.text;
        ctx.font      = 'bold 10px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(d.comparators, bx + bw/2, by - 6);

        // Depth
        ctx.fillStyle = C.muted; ctx.font = '8px Inter, system-ui';
        ctx.fillText('d=' + d.depth, bx + bw/2, by + 14);

        // Optimal badge
        if (d.optimal) {
          ctx.fillStyle = C.accent3; ctx.font = 'bold 7px Inter, system-ui';
          ctx.fillText('✓ OPTIMAL', bx + bw/2, by - 17);
        }
        if (d.isLower) {
          ctx.fillStyle = C.muted; ctx.font = '7px Inter, system-ui';
          ctx.fillText('lower bound', bx + bw/2, by - 17);
        }
        if (d.isBest) {
          ctx.fillStyle = C.accent2; ctx.font = 'bold 7px Inter, system-ui';
          ctx.fillText('BEST KNOWN', bx + bw/2, by - 17);
        }

        // X labels (two lines)
        const parts = d.label.split('\n');
        ctx.fillStyle = C.muted; ctx.font = '8.5px Inter, system-ui'; ctx.textAlign = 'center';
        parts.forEach((p, pi) => {
          ctx.fillText(p, bx + bw/2, padT + chartH + 14 + pi * 12);
        });
      });

      // Bracket annotation for n=16 gap
      const lowerIdx = 3, bestIdx = 4, batcherIdx = 5;
      const getLane  = idx => padL + idx * barW + barPad + (barW - barPad*2)/2;
      const xL = getLane(lowerIdx), xB = getLane(bestIdx);
      const yBracket = padT + 8;

      ctx.strokeStyle = C.accent2 + '99'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xL, yBracket+6); ctx.lineTo(xL, yBracket);
      ctx.lineTo(xB, yBracket);   ctx.lineTo(xB, yBracket+6);
      ctx.stroke();
      ctx.fillStyle = C.accent2; ctx.font = 'bold 9px Inter, system-ui'; ctx.textAlign = 'center';
      ctx.fillText('gap: 7', (xL + xB) / 2, yBracket - 3);
    }

    window.addEventListener('resize', draw);
    draw();
  })();

  // ── Scroll fade-in ─────────────────────────────────────────────────────────
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity  = '1';
        e.target.style.transform = 'none';
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('section').forEach(s => {
    s.style.opacity   = '0';
    s.style.transform = 'translateY(20px)';
    s.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    obs.observe(s);
  });

})();
