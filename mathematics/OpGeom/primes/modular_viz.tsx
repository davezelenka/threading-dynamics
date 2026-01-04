import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Play, Pause, SkipBack, SkipForward, Info, ZoomIn, ZoomOut, Crosshair } from 'lucide-react';

const ModularCoherenceViz = () => {
  const [position, setPosition] = useState(1);
  const [windowWidth, setWindowWidth] = useState(200);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [mousePos, setMousePos] = useState(null);
  const [pinnedPoint, setPinnedPoint] = useState(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  const END = 10000;
  const MODULI = [2, 3, 5, 7, 11];
  const COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'];
  
  // Optimized prime and omega calculation
  const { primeSet, omegaCache } = useMemo(() => {
    const sieve = (limit) => {
      const isPrime = new Array(limit + 1).fill(true);
      isPrime[0] = isPrime[1] = false;
      for (let i = 2; i * i <= limit; i++) {
        if (isPrime[i]) {
          for (let j = i * i; j <= limit; j += i) {
            isPrime[j] = false;
          }
        }
      }
      return isPrime;
    };

    const isPrimeArray = sieve(END);
    const primes = new Set();
    for (let i = 0; i <= END; i++) {
      if (isPrimeArray[i]) primes.add(i);
    }

    const omega = (n) => {
      if (n <= 1) return 0;
      let count = 0;
      let temp = n;
      for (let p = 2; p * p <= temp; p++) {
        while (temp % p === 0) {
          count++;
          temp /= p;
        }
      }
      if (temp > 1) count++;
      return count;
    };

    const cache = {};
    for (let i = 1; i <= END; i++) {
      cache[i] = primes.has(i) ? 0 : omega(i);
    }

    return { primeSet: primes, omegaCache: cache };
  }, []);

  // Handle mouse movement for coordinate tracking
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { top: 60, right: 40, bottom: 60, left: 60 };
    const plotWidth = rect.width - padding.left - padding.right;
    const plotHeight = rect.height - padding.top - padding.bottom;

    // Convert pixel coordinates to data coordinates
    const windowStart = Math.floor(position);
    const windowEnd = windowStart + windowWidth;
    const maxOmega = 18;

    if (x >= padding.left && x <= rect.width - padding.right &&
        y >= padding.top && y <= rect.height - padding.bottom) {
      
      const n = Math.round(windowStart + ((x - padding.left) / plotWidth) * windowWidth);
      const omega = (padding.top + plotHeight - y) / plotHeight * maxOmega;

      setMousePos({ x, y, n, omega: omega.toFixed(2) });
    } else {
      setMousePos(null);
    }
  };

  const handleClick = (e) => {
    if (mousePos) {
      setPinnedPoint({ n: mousePos.n, omega: mousePos.omega });
    }
  };

  // Draw visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 60, right: 40, bottom: 60, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear with gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
    bgGradient.addColorStop(0, '#0a0e27');
    bgGradient.addColorStop(1, '#1a1f3a');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    const windowStart = Math.floor(position);
    const windowEnd = windowStart + windowWidth;
    
    // Grid with glow
    ctx.strokeStyle = 'rgba(100, 150, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = padding.top + (plotHeight / 10) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw modular patterns
    const maxOmega = 18;

    MODULI.forEach((mod, idx) => {
      const color = COLORS[idx];
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowBlur = 15;
      ctx.shadowColor = color;

      const points = [];
      for (let n = windowStart; n < windowEnd; n++) {
        if (n % mod === 0 && n <= END) {
          const omega = omegaCache[n] || 0;
          const x = padding.left + ((n - windowStart) / windowWidth) * plotWidth;
          const y = padding.top + plotHeight - (omega / maxOmega) * plotHeight;
          points.push({ x, y, n, omega });
        }
      }

      // Draw connecting lines
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
    });

    // Draw prime indicators
    ctx.strokeStyle = 'rgba(255, 50, 80, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    for (let n = windowStart; n < windowEnd; n++) {
      if (primeSet.has(n)) {
        const x = padding.left + ((n - windowStart) / windowWidth) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // Axes and labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Text labels
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Number n', width / 2, height - 20);
    
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Ω(n)', 0, 0);
    ctx.restore();

    // Window range
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.fillStyle = '#4ECDC4';
    ctx.textAlign = 'center';
    ctx.fillText(`[${windowStart}, ${windowEnd}] • Width: ${windowWidth}`, width / 2, 40);

    // X-axis numbers
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    const numTicks = Math.min(10, Math.floor(windowWidth / 20));
    for (let i = 0; i <= numTicks; i++) {
      const n = Math.floor(windowStart + (windowWidth / numTicks) * i);
      const x = padding.left + (plotWidth / numTicks) * i;
      ctx.fillText(n.toString(), x, height - padding.bottom + 25);
    }

    // Draw crosshair and coordinates if mouse is over canvas
    if (mousePos) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      
      // Vertical line
      ctx.beginPath();
      ctx.moveTo(mousePos.x, padding.top);
      ctx.lineTo(mousePos.x, height - padding.bottom);
      ctx.stroke();
      
      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(padding.left, mousePos.y);
      ctx.lineTo(width - padding.right, mousePos.y);
      ctx.stroke();
      
      ctx.setLineDash([]);
      
      // Coordinate label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(mousePos.x + 10, mousePos.y - 30, 100, 25);
      ctx.fillStyle = '#4ECDC4';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`n=${mousePos.n}`, mousePos.x + 15, mousePos.y - 18);
      ctx.fillText(`Ω=${mousePos.omega}`, mousePos.x + 15, mousePos.y - 8);
    }

    // Draw pinned point marker
    if (pinnedPoint) {
      const n = parseInt(pinnedPoint.n);
      if (n >= windowStart && n < windowEnd) {
        const x = padding.left + ((n - windowStart) / windowWidth) * plotWidth;
        const omega = omegaCache[n] || 0;
        const y = padding.top + plotHeight - (omega / maxOmega) * plotHeight;
        
        // Draw crosshair
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(x - 15, y);
        ctx.lineTo(x + 15, y);
        ctx.moveTo(x, y - 15);
        ctx.lineTo(x, y + 15);
        ctx.stroke();
      }
    }

  }, [position, windowWidth, primeSet, omegaCache, mousePos, pinnedPoint]);

  // Auto-play animation
  useEffect(() => {
    if (isPlaying) {
      animationRef.current = setInterval(() => {
        setPosition(prev => {
          const next = prev + 2;
          if (next >= END - windowWidth) {
            setIsPlaying(false);
            return END - windowWidth;
          }
          return next;
        });
      }, 50);
    } else {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    }
    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    };
  }, [isPlaying, windowWidth]);

  const handleZoomIn = () => {
    setWindowWidth(prev => Math.max(50, Math.floor(prev / 1.5)));
  };

  const handleZoomOut = () => {
    setWindowWidth(prev => Math.min(END, Math.floor(prev * 1.5)));
  };

  return (
    <div className="w-full h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col">
      {/* Header */}
      <div className="p-4 text-center border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-1">
          Arithmetic Ecology Explorer
        </h1>
        <p className="text-gray-300 text-xs">
          Constraint-dissipation dynamics • Click to pin coordinates
        </p>
      </div>

      {/* Canvas */}
      <div className="flex-1 p-6 relative">
        <canvas
          ref={canvasRef}
          className="w-full h-full rounded-lg shadow-2xl cursor-crosshair"
          style={{ background: 'transparent' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setMousePos(null)}
          onClick={handleClick}
        />
        
        {/* Pinned point info */}
        {pinnedPoint && (
          <div className="absolute top-8 left-8 bg-black/90 backdrop-blur-md p-4 rounded-lg border border-yellow-500/50 shadow-2xl max-w-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-yellow-400" />
                <h3 className="text-yellow-400 font-bold">Analysis</h3>
              </div>
              <button
                onClick={() => setPinnedPoint(null)}
                className="text-gray-400 hover:text-white transition-colors"
                title="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="space-y-2 text-sm">
              <div className="font-mono">
                <div className="text-cyan-300 font-bold">n = {pinnedPoint.n}</div>
                <div className="text-cyan-300">Ω(n) = {omegaCache[pinnedPoint.n]}</div>
              </div>
              
              <div className="border-t border-white/20 pt-2">
                <div className="text-xs text-gray-400 mb-1">Status:</div>
                <div className={primeSet.has(parseInt(pinnedPoint.n)) ? 'text-green-400' : 'text-orange-400'}>
                  {primeSet.has(parseInt(pinnedPoint.n)) ? '✓ Prime (irreducible)' : '✗ Composite (constraint sink)'}
                </div>
              </div>

              <div className="border-t border-white/20 pt-2">
                <div className="text-xs text-gray-400 mb-1">Modular structure:</div>
                <div className="text-gray-300 text-xs">
                  {(() => {
                    const divisors = MODULI.filter(m => parseInt(pinnedPoint.n) % m === 0);
                    if (divisors.length === 0) return 'Coprime to {2,3,5,7,11}';
                    return `Divisible by: ${divisors.join(', ')}`;
                  })()}
                </div>
              </div>

              <div className="border-t border-white/20 pt-2">
                <div className="text-xs text-gray-400 mb-1">Prime desert context:</div>
                <div className="text-gray-300 text-xs">
                  {(() => {
                    const n = parseInt(pinnedPoint.n);
                    let prevPrime = n - 1;
                    let nextPrime = n + 1;
                    
                    while (prevPrime > 0 && !primeSet.has(prevPrime)) prevPrime--;
                    while (nextPrime <= END && !primeSet.has(nextPrime)) nextPrime++;
                    
                    const gapBefore = n - prevPrime;
                    const gapAfter = nextPrime - n;
                    const totalGap = nextPrime - prevPrime;
                    
                    if (primeSet.has(n)) {
                      return `Prime itself • Gaps: -${gapBefore}/+${gapAfter}`;
                    }
                    return `Gap: ${totalGap} (${prevPrime}...${nextPrime})`;
                  })()}
                </div>
              </div>

              <div className="border-t border-white/20 pt-2">
                <div className="text-xs text-gray-400 mb-1">Spike character:</div>
                <div className="text-gray-300 text-xs">
                  {(() => {
                    const n = parseInt(pinnedPoint.n);
                    const omega = omegaCache[n];
                    const divisors = MODULI.filter(m => n % m === 0);
                    
                    if (omega >= 6) {
                      return `🔥 High constraint (Ω=${omega}) • ${divisors.length} moduli converge`;
                    } else if (omega >= 4) {
                      return `⚡ Medium constraint • ${divisors.length} moduli active`;
                    } else if (omega === 0) {
                      return `✨ Irreducible • No constraint accumulation`;
                    } else {
                      return `Low constraint • ${divisors.length} moduli`;
                    }
                  })()}
                </div>
              </div>

              <div className="border-t border-white/20 pt-2">
                <div className="text-xs text-gray-400 mb-1">Factorization:</div>
                <div className="text-gray-300 text-xs font-mono">
                  {(() => {
                    const n = parseInt(pinnedPoint.n);
                    if (primeSet.has(n)) return `${n} (prime)`;
                    
                    // Factor the number
                    let temp = n;
                    const factors = [];
                    for (let p = 2; p <= temp; p++) {
                      let count = 0;
                      while (temp % p === 0) {
                        count++;
                        temp /= p;
                      }
                      if (count > 0) {
                        factors.push(count === 1 ? `${p}` : `${p}^${count}`);
                      }
                      if (temp === 1) break;
                    }
                    return factors.join(' × ');
                  })()}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info overlay */}
        {showInfo && (
          <div className="absolute top-8 right-8 bg-black/90 backdrop-blur-md p-6 rounded-lg border border-cyan-500/30 max-w-md shadow-2xl">
            <h3 className="text-cyan-400 font-bold mb-3 text-lg">Ecological View</h3>
            <div className="space-y-2 text-sm">
              {MODULI.map((mod, idx) => (
                <div key={mod} className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: COLORS[idx], boxShadow: `0 0 10px ${COLORS[idx]}` }} />
                  <span className="text-gray-300">n ≡ 0 (mod {mod})</span>
                </div>
              ))}
              <div className="flex items-center gap-2 pt-2 border-t border-white/10">
                <div className="w-4 h-0.5 bg-red-500/60" style={{ borderTop: '2px dashed' }} />
                <span className="text-gray-300">Prime numbers</span>
              </div>
            </div>
            <p className="text-gray-400 text-xs mt-4 leading-relaxed">
              Each thread shows constraint gradients for numbers divisible by that modulus. 
              Height = Ω(n) = number of prime factors with multiplicity.
              Hover to see exact coordinates. Click to pin a point.
            </p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-6 bg-black/30 backdrop-blur-md border-t border-white/10">
        <div className="max-w-5xl mx-auto space-y-4">
          {/* Slider */}
          <div className="flex items-center gap-4">
            <span className="text-cyan-400 font-mono text-sm min-w-16">{Math.floor(position)}</span>
            <input
              type="range"
              min="1"
              max={END - windowWidth}
              value={position}
              onChange={(e) => setPosition(Number(e.target.value))}
              className="flex-1 h-2 bg-gradient-to-r from-purple-900 to-cyan-900 rounded-full appearance-none cursor-pointer"
            />
            <span className="text-cyan-400 font-mono text-sm min-w-16">{END - windowWidth}</span>
          </div>

          {/* Buttons */}
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <button
              onClick={handleZoomOut}
              className="p-3 bg-purple-600/20 hover:bg-purple-600/40 border border-purple-500/30 rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/20"
              title="Zoom out (increase window width)"
            >
              <ZoomOut className="w-5 h-5 text-purple-300" />
            </button>

            <button
              onClick={handleZoomIn}
              className="p-3 bg-purple-600/20 hover:bg-purple-600/40 border border-purple-500/30 rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/20"
              title="Zoom in (decrease window width)"
            >
              <ZoomIn className="w-5 h-5 text-purple-300" />
            </button>
            
            <button
              onClick={() => setPosition(Math.max(1, position - 1000))}
              className="p-3 bg-purple-600/20 hover:bg-purple-600/40 border border-purple-500/30 rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/20"
            >
              <SkipBack className="w-5 h-5 text-purple-300" />
            </button>
            
            <button
              onClick={() => setPosition(Math.max(1, position - 100))}
              className="px-6 py-3 bg-cyan-600/20 hover:bg-cyan-600/40 border border-cyan-500/30 rounded-lg transition-all duration-200 text-cyan-300 font-semibold hover:shadow-lg hover:shadow-cyan-500/20"
            >
              -100
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-4 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 rounded-full transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105"
            >
              {isPlaying ? (
                <Pause className="w-6 h-6 text-white" />
              ) : (
                <Play className="w-6 h-6 text-white" />
              )}
            </button>

            <button
              onClick={() => setPosition(Math.min(END - windowWidth, position + 100))}
              className="px-6 py-3 bg-cyan-600/20 hover:bg-cyan-600/40 border border-cyan-500/30 rounded-lg transition-all duration-200 text-cyan-300 font-semibold hover:shadow-lg hover:shadow-cyan-500/20"
            >
              +100
            </button>

            <button
              onClick={() => setPosition(Math.min(END - windowWidth, position + 1000))}
              className="p-3 bg-purple-600/20 hover:bg-purple-600/40 border border-purple-500/30 rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/20"
            >
              <SkipForward className="w-5 h-5 text-purple-300" />
            </button>

            <button
              onClick={() => setShowInfo(!showInfo)}
              className={`p-3 ${showInfo ? 'bg-cyan-600/40' : 'bg-gray-600/20'} hover:bg-cyan-600/40 border border-cyan-500/30 rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-cyan-500/20 ml-4`}
            >
              <Info className="w-5 h-5 text-cyan-300" />
            </button>
          </div>

          {/* Quick jump and legend */}
          <div className="flex items-center justify-center gap-4 text-sm flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Jump:</span>
              {[7776, 8400].map(val => (
                <button
                  key={val}
                  onClick={() => setPosition(Math.max(1, Math.min(val - 50, END - windowWidth)))}
                  className="px-3 py-1 bg-orange-600/20 hover:bg-orange-600/40 border border-orange-500/30 rounded text-orange-300 transition-all duration-200 font-mono text-xs font-bold"
                >
                  {val}
                </button>
              ))}
              {[1000, 5000, 8190, 9210, 9400].map(val => (
                <button
                  key={val}
                  onClick={() => setPosition(Math.max(1, Math.min(val - 50, END - windowWidth)))}
                  className="px-3 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-gray-300 transition-all duration-200 font-mono text-xs"
                >
                  {val}
                </button>
              ))}
            </div>
            
            <div className="h-4 w-px bg-gray-600"></div>
            
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Moduli:</span>
              {MODULI.map((mod, idx) => (
                <div key={mod} className="flex items-center gap-1.5">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ 
                      backgroundColor: COLORS[idx], 
                      boxShadow: `0 0 8px ${COLORS[idx]}` 
                    }} 
                  />
                  <span className="text-gray-300 font-mono text-xs">{mod}</span>
                </div>
              ))}
              <div className="flex items-center gap-1.5 ml-2">
                <div className="w-3 h-0.5 bg-red-500/60" style={{ borderTop: '2px dashed' }} />
                <span className="text-gray-300 text-xs">Prime</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #4ECDC4, #45B7D1);
          cursor: pointer;
          box-shadow: 0 0 20px rgba(78, 205, 196, 0.6);
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #4ECDC4, #45B7D1);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 20px rgba(78, 205, 196, 0.6);
        }
      `}</style>
    </div>
  );
};

export default ModularCoherenceViz;
