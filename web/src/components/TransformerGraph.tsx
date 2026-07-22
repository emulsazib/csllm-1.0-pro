/** The Transformer stack in 3D, driven by REAL captured attention.
 *
 *  Layout — deliberately a readable encoding, not decoration:
 *    · X = key position in the sequence
 *    · Y = layer (0 at the bottom, the tied lm_head on top)
 *    · cell colour = how much that layer attends to that key, averaged over its
 *      selected heads
 *    · arcs = the currently selected layer's per-head attention from the newest
 *      token back to each key, width and opacity following the weight
 *
 *  Attention is a magnitude in [0,1] with no polarity, so the cells use a
 *  SEQUENTIAL one-hue ramp (light→dark on a light surface, dark→light on a dark
 *  one). A diverging ramp would invent a midpoint that means nothing here.
 */

import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { type AttentionBlock, attentionByLayerKey, attentionRow } from "../api/ws";
import { type Mode, currentMode, cssVar, sequentialColor } from "../theme";

/** The ramp itself lives in theme.ts so the 3D view and the 2D heat-map cannot
 *  drift apart; this only wraps it in the type three.js wants. */
function sequential(t: number, mode: Mode): THREE.Color {
  return new THREE.Color(sequentialColor(t, mode));
}

const CELL = 0.42;
const GAP = 0.1;

interface SceneProps {
  block: AttentionBlock;
  labels: string[];
  layerIndex: number;
  headIndex: number | "all";
  mode: Mode;
  pulse: number;
}

function AttentionGrid({ block, layerIndex, headIndex, mode, pulse }: SceneProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = block.layers * block.keys;

  // Per-layer key attention, computed once per frame payload rather than per
  // render tick — this is O(layers × heads × keys).
  const perLayer = useMemo(
    () => Array.from({ length: block.layers }, (_, l) => attentionByLayerKey(block, l)),
    [block],
  );
  const peak = useMemo(
    () => Math.max(1e-6, ...perLayer.map((row) => Math.max(...row))),
    [perLayer],
  );

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    let i = 0;
    for (let layer = 0; layer < block.layers; layer++) {
      for (let key = 0; key < block.keys; key++) {
        const weight = perLayer[layer][key] / peak;
        dummy.position.set(
          (key - block.keys / 2) * (CELL + GAP),
          layer * (CELL + GAP) * 1.6,
          0,
        );
        // Height carries magnitude too, so the encoding survives greyscale and
        // colour-vision deficiency.
        dummy.scale.set(CELL, CELL, 0.08 + weight * 0.9);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        mesh.setColorAt(i, sequential(weight, mode));
        i++;
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [block, perLayer, peak, mode, count]);

  // A gentle drift so the stack reads as three-dimensional; the pulse nudges it
  // when a new token lands.
  const groupRef = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.18) * 0.22;
    }
  });

  const arcs = useMemo(() => {
    if (headIndex === "all") return [];
    const row = attentionRow(block, layerIndex, headIndex);
    const rowPeak = Math.max(1e-6, ...row);
    const originX = (block.keys - 1 - block.keys / 2) * (CELL + GAP);
    const y = layerIndex * (CELL + GAP) * 1.6;
    return Array.from(row, (weight, key) => ({
      key,
      weight: weight / rowPeak,
      points: [
        new THREE.Vector3(originX, y, 0.6),
        new THREE.Vector3(
          ((key + block.keys - 1) / 2 - block.keys / 2) * (CELL + GAP),
          y + 1.1 + weight * 2.2,
          0.6,
        ),
        new THREE.Vector3((key - block.keys / 2) * (CELL + GAP), y, 0.6),
      ],
    }));
  }, [block, layerIndex, headIndex]);

  return (
    <group ref={groupRef} scale={pulse}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]} key={count}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.55} metalness={0.05} />
      </instancedMesh>

      {arcs.map((arc) =>
        arc.weight < 0.04 ? null : (
          <Arc key={arc.key} points={arc.points} weight={arc.weight} mode={mode} />
        ),
      )}
    </group>
  );
}

function Arc({
  points,
  weight,
  mode,
}: {
  points: THREE.Vector3[];
  weight: number;
  mode: Mode;
}) {
  const geometry = useMemo(() => {
    const curve = new THREE.QuadraticBezierCurve3(points[0], points[1], points[2]);
    return new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
  }, [points]);

  return (
    <primitive
      object={
        new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({
            color: sequential(weight, mode),
            transparent: true,
            opacity: 0.25 + weight * 0.7,
          }),
        )
      }
    />
  );
}

export function TransformerGraph({
  block,
  labels,
  layerIndex,
  headIndex,
}: {
  block: AttentionBlock | null;
  labels: string[];
  layerIndex: number;
  headIndex: number | "all";
}) {
  const mode = currentMode();
  const [pulse, setPulse] = useState(1);

  // Brief scale bump each time a new attention block arrives, so the eye is
  // drawn to the update rather than having to spot it.
  useEffect(() => {
    if (!block) return;
    setPulse(1.035);
    const timer = setTimeout(() => setPulse(1), 140);
    return () => clearTimeout(timer);
  }, [block]);

  if (!block) {
    return <div className="empty">Run a prompt to see attention flow through the stack.</div>;
  }

  const surface = cssVar("--surface-1", "#fcfcfb");

  // Frame the stack rather than using a fixed camera: the block grows with both
  // the context length and the layer count, and a fixed distance leaves a short
  // wide grid stranded in the middle of the canvas.
  const width = block.keys * (CELL + GAP);
  const height = block.layers * (CELL + GAP) * 1.6;
  const distance = Math.max(7, Math.max(width * 0.82, height * 1.7));

  return (
    <div style={{ height: 340, borderRadius: 8, overflow: "hidden" }}>
      <Canvas camera={{ position: [0, height * 0.42, distance], fov: 42 }} dpr={[1, 2]}>
        <color attach="background" args={[surface]} />
        <ambientLight intensity={mode === "dark" ? 0.85 : 1.15} />
        <directionalLight position={[6, 10, 8]} intensity={mode === "dark" ? 1.1 : 1.4} />
        <group position={[0, -height / 2, 0]}>
          <AttentionGrid
            block={block}
            labels={labels}
            layerIndex={layerIndex}
            headIndex={headIndex}
            mode={mode}
            pulse={pulse}
          />
        </group>
      </Canvas>
    </div>
  );
}
